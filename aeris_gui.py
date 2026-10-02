import sys
import os
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

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

# --- ŚCIEŻKI SYSTEMOWE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "aeris_tokenizer_32k.model")
MODEL_PATH = os.path.join(BASE_DIR, "model", "aeris_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ŁADOWANIE TOKENIZERA I MODELU ---
GLOBAL_SP = None
GLOBAL_MODEL = None

if os.path.exists(TOKENIZER_PATH):
    GLOBAL_SP = spm.SentencePieceProcessor()
    GLOBAL_SP.load(TOKENIZER_PATH)

if os.path.exists(MODEL_PATH) and GLOBAL_SP is not None:
    config = Config()
    GLOBAL_MODEL = AERISTransformerModel(config)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    GLOBAL_MODEL.load_state_dict(state_dict, strict=False)
    GLOBAL_MODEL.to(DEVICE)
    GLOBAL_MODEL.eval()

# --- WĄTEK GENEROWANIA ---
class ModelWorker(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, chat_history):
        super().__init__()
        self.chat_history = chat_history

    @torch.no_grad()
    def run(self):
        if GLOBAL_MODEL is None or GLOBAL_SP is None:
            self.response_ready.emit("Błąd: Model lub tokenizer nie został załadowany.")
            return

        try:
            # 1. Budowanie promptu z wieloturowej historii dialogu
            formatted_dialogue = []
            for role, text in self.chat_history:
                formatted_dialogue.append(f"{role}: {text}")
            
            full_prompt = "\n".join(formatted_dialogue) + "\nAeris:"
            input_ids = GLOBAL_SP.encode(full_prompt, out_type=int)
            
            # Dynamiczny rozmiar kontekstu z configu z rezerwą 150 tokenów na generację
            max_seq_len = getattr(GLOBAL_MODEL.config, 'max_seq_len', getattr(GLOBAL_MODEL.config, 'block_size', 1024))
            max_context = max_seq_len - 150
            if len(input_ids) > max_context:
                input_ids = input_ids[-max_context:]

            x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            generated = []
            max_new_tokens = 100
            
            temperature = 0.52
            top_k = 28
            top_p = 0.82
            repetition_penalty = 1.25

            for _ in range(max_new_tokens):
                x_cond = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]
                logits = GLOBAL_MODEL(x_cond)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)

                for token_id in set(generated + input_ids[-20:]):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float('Inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = -float('Inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                if token_id == GLOBAL_SP.eos_id():
                    break

                x = torch.cat((x, next_token), dim=1)
                generated.append(token_id)

                text_so_far = GLOBAL_SP.decode(generated)
                if "\nUżytkownik:" in text_so_far or "<Użytkownik>:" in text_so_far:
                    break

            raw_text = GLOBAL_SP.decode(generated).strip()
            
            # Odcięcie ewentualnych halucynowanych wypowiedzi użytkownika
            for tag in ["\nUżytkownik:", "<Użytkownik>:", "Użytkownik:"]:
                if tag in raw_text:
                    raw_text = raw_text.split(tag)[0].strip()

            # Usunięcie powtórzonych prefiksów z początku odpowiedzi
            while raw_text.startswith("Aeris:") or raw_text.startswith("<Aeris>:"):
                raw_text = raw_text.replace("Aeris:", "", 1).replace("<Aeris>:", "", 1).strip()

            response_text = raw_text.rstrip('”"„')
            response_text = " ".join(response_text.split())
            
            # Domykanie do logicznego końca zdania
            valid_ends = ('.', '!', '?', '…', ':)')
            if not response_text.endswith(valid_ends):
                last_punct = max(response_text.rfind('.'), response_text.rfind('!'), response_text.rfind('?'))
                if last_punct != -1:
                    response_text = response_text[:last_punct+1]

            self.response_ready.emit(response_text)

        except Exception as e:
            self.response_ready.emit(f"Wystąpił błąd podczas generowania: {str(e)}")

# --- OKNO APLIKACJI ---
class AerisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aeris – Interfejs Dialogowy")
        self.resize(700, 600)

        self.chat_history = []  # Pamięć kontekstowa bieżącej sesji

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(20, 20, 20))
        palette.setColor(QPalette.WindowText, QColor(0, 235, 120))
        palette.setColor(QPalette.Base, QColor(15, 15, 15))
        palette.setColor(QPalette.Text, QColor(0, 235, 120))
        self.setPalette(palette)

        layout = QVBoxLayout()
        status_text = "🟢 Aeris gotowy do rozmowy." if GLOBAL_MODEL is not None else "⚠️ Brak wag modelu w folderze model/."
        status_color = "#00ff75" if GLOBAL_MODEL is not None else "#ffaa00"
        self.status_label = QLabel(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(self.status_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.append("=== AERIS TRANSFORMER – LOKALNY INTERFEJS ===\n")
        layout.addWidget(self.output)

        hbox = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Napisz wiadomość...")
        self.input.returnPressed.connect(self.handle_input)
        hbox.addWidget(self.input)

        self.send_button = QPushButton("Wyślij")
        self.send_button.clicked.connect(self.handle_input)
        hbox.addWidget(self.send_button)
        layout.addLayout(hbox)

        file_btn = QPushButton("📁 Otwórz plik danych / projektu")
        file_btn.clicked.connect(self.handle_open_file)
        layout.addWidget(file_btn)

        self.setLayout(layout)
        self.worker = None

    def handle_input(self):
        text = self.input.text().strip()
        if not text:
            return

        if GLOBAL_MODEL is None:
            QMessageBox.warning(self, "Brak modelu", "Nie znaleziono pliku wag 'model/aeris_model.pt'. Umieść wagi w odpowiednim katalogu.")
            return

        self.send_button.setEnabled(False)
        self.input.setEnabled(False)
        self.output.append(f"<b>Użytkownik:</b> {text}")
        self.input.clear()
        self.status_label.setText("⏳ Aeris przetwarza wypowiedź...")

        # Zapis do historii dialogu
        self.chat_history.append(("Użytkownik", text))

        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()

        self.worker = ModelWorker(self.chat_history)
        self.worker.response_ready.connect(self.on_response_ready)
        self.worker.start()

    def on_response_ready(self, response):
        self.output.append(f"<b>Aeris:</b> {response}\n")
        self.status_label.setText("🟢 Aeris gotowy.")
        self.send_button.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

        # Zapis odpowiedzi do historii dialogu
        self.chat_history.append(("Aeris", response))

    def handle_open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik projektu", BASE_DIR, "Pliki tekstowe i skrypty (*.txt *.py *.md)")
        if file_path:
            # Bezpieczne otwarcie pliku w domyślnym programie systemowym
            if sys.platform.startswith('win'):
                os.startfile(file_path)
            elif sys.platform.startswith('darwin'):
                subprocess.call(('open', file_path))
            else:
                subprocess.call(('xdg-open', file_path))

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        gui = AerisGUI()
        gui.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("❌ Błąd przy uruchamianiu GUI:", e)