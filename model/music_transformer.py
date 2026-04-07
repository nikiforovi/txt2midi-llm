import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class MusicTransformer(nn.Module):
    """Decoder-only Transformer for MIDI generation conditioned on prompt embeddings."""
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 256, 
        nhead: int = 8, 
        num_layers: int = 6, 
        dim_feedforward: int = 1024,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # We use the prompt embedding as a prefix or inject it via cross-attention.
        # For the baseline, we'll use it as a prefix (Prepended to the sequence).
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tokens, prompt_embedding):
        """
        Args:
            tokens: (batch_size, seq_len)
            prompt_embedding: (batch_size, d_model)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # For the baseline, we treat prompt_embedding as the "memory" in TransformerDecoder.
        # This means every decoder step attends to the prompt embedding.
        # Memory shape: (batch_size, 1, d_model)
        memory = prompt_embedding.unsqueeze(1)
        
        tgt_mask = self.generate_causal_mask(tokens.size(1)).to(tokens.device)
        
        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        logits = self.fc_out(output)
        return logits
