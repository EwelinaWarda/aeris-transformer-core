import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# Czyste importy architektury i konfiguracji z projektu
from config import Config
from model import AERISTransformerModel

# ==========================================
# 1. KONFIGURACJA ŚCIEŻEK I HIPERPARAMETRÓW
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_adapt")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model", "aeris_model.pt")
SAVE_PATH = os.path.join(BASE_DIR, "model", "aeris_model_adapted.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "aeris_tokenizer_32k.model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-6  # Konserwatywny learning rate zapobiegający catastrophic forgetting
BLOCK_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = 2

print(f"[*] Urządzenie: {DEVICE}")
print(f"[*] Checkpoint bazowy: {CHECKPOINT_PATH}")
print(f"[*] Folder danych docelowych: {DATA_DIR}")

# ==========================================
# 2. TOKENIZACJA I OBSŁUGA DANYCH
# ==========================================
sp = spm.SentencePieceProcessor()
if os.path.exists(TOKENIZER_PATH):
    sp.load(TOKENIZER_PATH)
else:
    raise FileNotFoundError(f"Nie odnaleziono tokenizera pod ścieżką: {TOKENIZER_PATH}")

# Wczytanie wszystkich plików tekstowych z folderu data_finetune/
txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
full_text = ""

for path in txt_files:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        full_text += content + "\n\n"
        print(f"[+] Wczytano dane z: {os.path.basename(path)} ({len(content)} znaków)")

if not full_text.strip():
    print("[!] Ostrzeżenie: Brak danych w folderze data_finetune/. Używam syntetycznego tekstu testowego.")
    full_text = "Aeris jest autorskim modelem językowym opartym na architekturze Decoder-only Transformer."

tokens = sp.encode(full_text, out_type=int)
print(f"[*] Łączna liczba stokenizowanych jednostek: {len(tokens)}")

class TextBlockDataset(Dataset):
    def __init__(self, token_list, block_size):
        self.samples = []
        step = max(1, block_size // 2)  # Okno przesuwne z 50% nakładaniem
        for i in range(0, len(token_list) - block_size, step):
            chunk = token_list[i : i + block_size + 1]
            if len(chunk) == block_size + 1:
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = TextBlockDataset(tokens, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"[*] Przygotowano {len(dataset)} sekwencji treningowych.")

# ==========================================
# 3. INICJALIZACJA MODELU I ŁADOWANIE CHECKPOINTU
# ==========================================
config = Config()
model = AERISTransformerModel(config)

if os.path.exists(CHECKPOINT_PATH):
    print(f"[*] Ładowanie wag bazowych z: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
else:
    print(f"[!] Checkpoint ({CHECKPOINT_PATH}) nie został znaleziony. Inicjalizacja z wagami losowymi.")

model.to(DEVICE)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. PĘTLA DALSZEGO TRENOWANIA (CONTINUAL TRAINING)
# ==========================================
print("\n" + "=" * 50)
print("[*] START PROCESU FINE-TUNINGU (DOMAIN ADAPTATION)")
print("=" * 50 + "\n")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
            print(f"Epoka [{epoch}/{EPOCHS}] | Krok [{step+1}/{len(dataloader)}] | Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")

    avg_loss = total_loss / max(1, len(dataloader))
    print(f"\n---> KONIEC EPOKI {epoch} | Średni Loss: {avg_loss:.4f}\n" + "-" * 40)

# ==========================================
# 5. ZAPIS WYNIKOWYCH WAG
# ==========================================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save({"model_state": model.state_dict()}, SAVE_PATH)
print(f"\n[V] Sukces: Model został zapisany w {SAVE_PATH}")