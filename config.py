# config.py

class Config:
    def __init__(self):
        self.vocab_size = 32000                  # liczba tokenów
        self.n_layers = 16                       # liczba warstw transformerów
        self.n_heads = 8                         # liczba głów uwagi
        self.embedding_dim = 768                 # rozmiar osadzania
        self.ffn_dim = 3072                      # rozmiar feed-forward
        self.max_seq_len = 1024                  # maksymalna długość sekwencji
        self.dropout = 0.1                       # domyślny dropout
        self.gradient_checkpointing = True       # oszczędność RAMu
        self.precision = "fp16"                  # precyzja obliczeń
        self.batch_size = 2                      # wielkość batcha
        self.device = "cpu"                      # urządzenie docelowe
        self.lr = 0.0005
        self.epochs = 10
        self.load_model = True
        self.checkpoint_path = "checkpoints/epoch_3.pt"
        self.start_epoch = 4
        self.eos_token_id = 2