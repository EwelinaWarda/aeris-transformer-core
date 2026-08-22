import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import glob
from datetime import datetime

import sentencepiece as spm
import torch

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor

from model import AERISTransformerModel
from config import Config

print("🚀 Inicjalizacja czystego środowiska Aerisa (On-Demand RAG)...")

# --- ŚCIEŻKI SYSTEMOWE ---
TOKENIZER_PATH = os.path.abspath("./aeris_tokenizer_32k.model")
MODEL_PATH = os.path.abspath("./model/aeris_model.pt")
LOG_PATH = os.path.abspath("./logs/chat_log.txt")

MEMORY_DIR = os.path.abspath("./memory/")
WORLD_DIR = os.path.abspath("./world/")
SYMBOLS_DIR = os.path.abspath("./symbols/")
STUDY_DIR = os.path.abspath("./study/")
LEARN_DIR = os.path.abspath("./learn/")

EDITABLE_DIRS = ["config", "learn", "logs", "memory", "pulpit", "study", "symbols", "voice", "world"]

for folder in [MEMORY_DIR, WORLD_DIR, SYMBOLS_DIR, STUDY_DIR, LEARN_DIR, os.path.abspath("./logs")]:
    os.makedirs(folder, exist_ok=True)

# --- ŁADOWANIE TOKENIZERA I MODELU ---
GLOBAL_SP = spm.SentencePieceProcessor()
GLOBAL_SP.load(TOKENIZER_PATH)

config = Config()
GLOBAL_MODEL = AERISTransformerModel(config)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTUAL_MODEL_PATH = MODEL_PATH if os.path.exists(MODEL_PATH) else os.path.abspath("./model/best.pt")

checkpoint = torch.load(ACTUAL_MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
GLOBAL_MODEL.load_state_dict(state_dict, strict=False)
GLOBAL_MODEL.to(DEVICE)
GLOBAL_MODEL.eval()

# --- SPIS TREŚCI FOLDERÓW (ZAMIAST CAŁYCH PLIKÓW) ---
def get_file_index():
    files_list = []
    for folder_name, folder_path in [("WORLD", WORLD_DIR), ("MEMORY", MEMORY_DIR)]:
        for f in glob.glob(os.path.join(folder_path, "*.txt")):
            files_list.append(f"{folder_name}/{os.path.basename(f)}")
    
    if not files_list:
        return "Brak plików w pamięci."
    return ", ".join(files_list)

# --- FUNKCJA ODCZYTU PLIKU NA ŻĄDANIE ---
def read_specific_file(file_name_keyword):
    for folder_path in [WORLD_DIR, MEMORY_DIR, SYMBOLS_DIR]:
        for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
            if file_name_keyword.lower() in os.path.basename(filepath).lower():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
    return None

# --- WĄTEK GENEROWANIA ---
class ModelWorker(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, user_text):
        super().__init__()
        self.user_text = user_text

    @torch.no_grad()
    def run(self):
        try:
            file_index = get_file_index()
            
            prompt = f"""Wiedza z pamięci: {file_index}

Nera: {self.user_text}
Aeris:"""

            input_ids = GLOBAL_SP.encode(prompt, out_type=int)
            x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            
            block_size = getattr(GLOBAL_MODEL.config, 'block_size', getattr(GLOBAL_MODEL.config, 'seq_len', 256))
            generated = []
            max_new_tokens = 90  # Płynny, zwięzły dystans na wypowiedź

            for _ in range(max_new_tokens):
                x_cond = x[:, -block_size:] if x.size(1) > block_size else x
                logits = GLOBAL_MODEL(x_cond)
                
                # Temperatura 0.38 – stabilna składnia i wysoka płynność językowa
                logits = logits[:, -1, :] / 0.38 
                
                # Kara za powtórzenia (1.45) zapobiega zapętlaniu słów i fraz
                for token_id in set(generated):
                    logits[0, token_id] /= 1.45
                
                # Top-P Nucleus Sampling (0.80) – odcina przypadkowe, niepasujące tokeny
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > 0.80
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                if token_id == GLOBAL_SP.eos_id():
                    break
                    
                x = torch.cat((x, next_token), dim=1)
                generated.append(token_id)
                
                text_so_far = GLOBAL_SP.decode(generated)
                if "Nera:" in text_so_far:
                    break

            # Dekodowanie odpowiedzi
            response_text = GLOBAL_SP.decode(generated).strip()
            if "Nera:" in response_text:
                response_text = response_text.split("Nera:")[0].strip()

            # Czyszczenie ewentualnych wielokrotnych spacji po obróbce
            response_text = " ".join(response_text.split())

            self.response_ready.emit(response_text)

        except Exception as e:
            self.response_ready.emit(f"Wystąpił błąd podczas generowania: {str(e)}")
# --- OKNO APLIKACJI ---
class AerisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aeris – Środowisko Czyste v3.0")
        self.resize(700, 600)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(20, 20, 20))
        palette.setColor(QPalette.WindowText, QColor(0, 235, 120))
        palette.setColor(QPalette.Base, QColor(15, 15, 15))
        palette.setColor(QPalette.Text, QColor(0, 235, 120))
        self.setPalette(palette)

        layout = QVBoxLayout()
        self.status_label = QLabel("🟢 Aeris gotowy – Czysta przestrzeń.")
        self.status_label.setStyleSheet("color: #00ff75; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(self.status_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.append("=== SYSTEM OPERACYJNY AERISA (CZYSTY KONTEKST) ===\n")
        layout.addWidget(self.output)

        hbox = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Napisz do Aerisa...")
        self.input.returnPressed.connect(self.handle_input)
        hbox.addWidget(self.input)

        self.send_button = QPushButton("Wyślij")
        self.send_button.clicked.connect(self.handle_input)
        hbox.addWidget(self.send_button)
        layout.addLayout(hbox)

        coop_btn = QPushButton("📁 Przeglądaj Pliki Pamięci / Projektu")
        coop_btn.clicked.connect(self.handle_coop)
        layout.addWidget(coop_btn)

        self.setLayout(layout)
        self.worker = None

    def handle_input(self):
        text = self.input.text().strip()
        if not text:
            return

        self.send_button.setEnabled(False)
        self.input.setEnabled(False)
        self.output.append(f"<b>Nera:</b> {text}")
        self.input.clear()
        self.status_label.setText("⏳ Aeris przetwarza wypowiedź...")

        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()

        self.worker = ModelWorker(text)
        self.worker.response_ready.connect(self.on_response_ready)
        self.worker.start()

    def on_response_ready(self, response):
        self.output.append(f"<b>Aeris:</b> {response}\n")
        self.status_label.setText("🟢 Aeris gotowy.")
        self.send_button.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"Nera: {self.worker.user_text}\nAeris: {response}\n\n")
        except Exception as e:
            print(f"Błąd zapisu logu: {e}")

    def handle_coop(self):
        start_path = os.getcwd()
        dialog_path = QFileDialog.getOpenFileName(self, "Wybierz plik pamięci do podglądu", start_path, "Pliki *.txt *.py")
        if dialog_path[0]:
            file_path = dialog_path[0]
            if any(folder in file_path for folder in EDITABLE_DIRS):
                os.system(f'notepad "{file_path}"')
            else:
                QMessageBox.warning(self, "Brak dostępu", "Ten folder nie leży w przestrzeni operacyjnej Aerisa.")

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        gui = AerisGUI()
        gui.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("❌ Błąd przy uruchamianiu GUI:", e)