import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embedding import TransformerEmbedding
from config import Config


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.linear1 = nn.Linear(config.embedding_dim, config.ffn_dim)
        self.linear2 = nn.Linear(config.ffn_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x, attn_mask=None):
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(
            norm_x, norm_x, norm_x, 
            attn_mask=attn_mask,
            is_causal=False
        )
        x = x + self.dropout(attn_output)

        norm_x2 = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(norm_x2))))
        x = x + self.dropout(ff_output)
        return x


class TransformerStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.final_norm(x)


class AERISTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = TransformerEmbedding(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            max_seq_len=self.config.max_seq_len
        )
        self.transformer = TransformerStack(self.config)
        self.output_head = nn.Linear(self.config.embedding_dim, self.config.vocab_size, bias=False)

        # Inicjalizacja wag
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x)  # [B, T, C]
        T = x.size(1)
        
        # Prawidłowa trójkątna maska Causal Mask (minus nieskończoność na górnym trójkącie)
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        
        x = self.transformer(x, attn_mask=causal_mask)
        return self.output_head(x)

    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=40):
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            idx_cond = generated if generated.size(1) <= self.config.max_seq_len else generated[:, -self.config.max_seq_len:]
            
            with torch.no_grad():
                logits = self.forward(idx_cond)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat((generated, next_token), dim=1)

                if hasattr(self.config, 'eos_token_id') and next_token.item() == self.config.eos_token_id:
                    break

        return generated