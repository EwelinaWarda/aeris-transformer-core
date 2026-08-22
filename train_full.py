import os
import sys
import glob
import time
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import AERISTransformerModel
from config import Config
import sentencepiece as spm

class AerisTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


@dataclass
class TrainCfg:
    symbolic_dir: str = "./data"
    val_path: str | None = None
    spm_model_path: str = "aeris_tokenizer_32k.model"
    
    
    base_ckpt_path: str | None = None
    
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Zoptymalizowana konfiguracja SFT
    epochs: int = 15
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 5.0e-4                   
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05

    # Dane
    seq_len: int = 1024
    stride: int = 1024

    # Techniczne
    seed: int = 1337
    num_threads: int = 8


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def list_train_files(symbolic_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(symbolic_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików .txt w {symbolic_dir}")
    return files


def load_and_mask_data(file_paths: list[str], tokenizer: AerisTokenizer, seq_len: int, stride: int):
    print(f"[INFO] Błyskawiczne ładowanie PEŁNEGO zbioru ({len(file_paths)} plików)...")
    
    all_ids = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        all_ids.extend(tokenizer.sp.encode(text, out_type=int))

    all_chunks_x = []
    all_chunks_y = []

    for start in range(0, len(all_ids) - (seq_len + 1), stride):
        chunk = all_ids[start : start + seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        all_chunks_x.append(x)
        all_chunks_y.append(y)

    print(f"[INFO] Sukces! Przygotowano {len(all_chunks_x):,} okien sekwencji z łącznej liczby {len(all_ids):,} tokenów!")
    return torch.stack(all_chunks_x), torch.stack(all_chunks_y)


@torch.no_grad()
def evaluate(model, val_x, val_y, device, batch_size=4) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, val_x.size(0), batch_size):
        x_b = val_x[i : i + batch_size].to(device)
        y_b = val_y[i : i + batch_size].to(device)

        logits = model(x_b)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y_b.view(B * T))

        active_tokens = (y_b != -100).sum().item()
        if active_tokens > 0:
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens

    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def save_ckpt(path, model, optim, epoch, val_loss, best_val):
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val": best_val,
    }, path)


def main():
    cfg = TrainCfg()

    torch.set_num_threads(cfg.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    base_cfg = Config()
    cfg.seq_len = base_cfg.max_seq_len

    tokenizer = AerisTokenizer(cfg.spm_model_path)

    train_files = list_train_files(cfg.symbolic_dir)
    train_x, train_y = load_and_mask_data(train_files, tokenizer, cfg.seq_len, cfg.stride)
    
    val_x, val_y = None, None
    if os.path.exists(cfg.val_path):
        val_x, val_y = load_and_mask_data([cfg.val_path], tokenizer, cfg.seq_len, cfg.stride)

    # Inicjalizacja modelu
    model = AERISTransformerModel(base_cfg).to(device)

    # -------------------------------------------------------------
    # WCZYTYWANIE WAG BAZOWYCH (BEZ STAREGO OPTYMALIZATORA)
    # -------------------------------------------------------------
    if cfg.base_ckpt_path and os.path.exists(cfg.base_ckpt_path):
        print(f"\n[INFO] 🔄 Wczytuję wagi bazowe z checkpointu: {cfg.base_ckpt_path}")
        checkpoint = torch.load(cfg.base_ckpt_path, map_location=device)
        
        # Obsługa wariantów zapisu w pytorch
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print("[INFO] ✅ Wagi wczytane pomyślnie! Startujemy z wyższego poziomu wiedzy.\n")
    else:
        print("\n[INFO] ⚠️ Brak pliku base_ckpt_path lub plik nie istnieje. Start od całkowitego zera!\n")

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and p.ndim >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or p.ndim < 2)]

    optim = AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=cfg.lr)

    start_epoch = 1
    best_val = float("inf")

    steps_per_epoch = max(1, train_x.size(0) // cfg.batch_size)
    total_steps = max(1, (steps_per_epoch // cfg.grad_accum_steps) * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    log_path = os.path.join(cfg.log_dir, "train.log")
    global_step = 0

    print(f"[INFO] ROZPOCZYNAMY TRENING! Liczba kroków w epoce: {steps_per_epoch} | Docelowy LR: {cfg.lr}\n")

    with open(log_path, "a", encoding="utf-8") as logf:
        for epoch in range(start_epoch, cfg.epochs + 1):
            model.train()

            perm = torch.randperm(train_x.size(0))
            train_x_shuffled = train_x[perm]
            train_y_shuffled = train_y[perm]

            epoch_loss_sum = 0.0
            epoch_tokens = 0
            optim.zero_grad(set_to_none=True)

            micro_step = 0

            for i in range(0, train_x.size(0), cfg.batch_size):
                x_b = train_x_shuffled[i : i + cfg.batch_size].to(device)
                y_b = train_y_shuffled[i : i + cfg.batch_size].to(device)

                if x_b.size(0) < cfg.batch_size:
                    break

                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    logits = model(x_b)
                    B, T, V = logits.shape
                    loss = loss_fn(logits.view(B * T, V), y_b.view(B * T))
                    loss_scaled = loss / cfg.grad_accum_steps

                loss_scaled.backward()

                active_tokens = (y_b != -100).sum().item()
                if active_tokens > 0:
                    epoch_loss_sum += loss.item() * active_tokens
                    epoch_tokens += active_tokens

                micro_step += 1

                if micro_step % cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                    if global_step < warmup_steps:
                        curr_lr = cfg.lr * (global_step / warmup_steps)
                        for param_group in optim.param_groups:
                            param_group['lr'] = curr_lr
                    else:
                        scheduler.step()
                        curr_lr = scheduler.get_last_lr()[0]
                        for param_group in optim.param_groups:
                            param_group['lr'] = curr_lr

                    optim.zero_grad(set_to_none=True)
                    global_step += 1

            train_loss = epoch_loss_sum / max(1, epoch_tokens)
            val_loss = evaluate(model, val_x, val_y, device) if val_x is not None else train_loss

            ckpt_epoch = os.path.join(cfg.ckpt_dir, f"epoch_{epoch}.pt")
            save_ckpt(ckpt_epoch, model, optim, epoch, val_loss, best_val)

            if val_loss < best_val:
                best_val = val_loss
                ckpt_best = os.path.join(cfg.ckpt_dir, "best.pt")
                save_ckpt(ckpt_best, model, optim, epoch, val_loss, best_val)

            current_lr = optim.param_groups[0]['lr']
            msg = (
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_val={best_val:.6f} | lr={current_lr:.2e}\n"
            )
            print(msg.strip())
            logf.write(msg)
            logf.flush()


if __name__ == "__main__":
    main()