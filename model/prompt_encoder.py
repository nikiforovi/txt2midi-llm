import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, List

class PromptEncoderV2(nn.Module):
    """v2: Multi-factor conditioning (Text + Tempo + Mode + Chromaticity)."""
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny", embedding_dim: int = 256):
        super().__init__()
        # 1. Text (Frozen BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_out_dim = self.bert.config.hidden_size
        
        # 2. Mode (Categorical)
        self.mode_list = [
            "major", "minor", "dorian", "phrygian", "lydian", 
            "mixolydian", "locrian", "dorian_b2", "lydian_dominant"
        ]
        self.mode_embedding = nn.Embedding(len(self.mode_list) + 1, 64)
        
        # 3. Numerical (Scalar projection)
        # Inputs: [Tempo, Chromaticity]
        self.numerical_projection = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 4. Fusion Layer
        # bert + mode + numerical -> model_dim
        self.fusion = nn.Sequential(
            nn.Linear(bert_out_dim + 64 + 64, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, prompts: List[str], modes: List[str], tempos: torch.Tensor, chromaticities: torch.Tensor):
        """
        Args:
            prompts: List of text descriptions
            modes: List of mode names
            tempos: tensor of (batch_size, 1) - normalized 0-1 (e.g. bpm/300)
            chromaticities: tensor of (batch_size, 1) - 0-1
        """
        device = tempos.device
        
        # Text embedding
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        text_out = self.bert(**inputs).last_hidden_state[:, 0, :]
        
        # Mode embedding
        mode_ids = []
        for m in modes:
            if m in self.mode_list:
                mode_ids.append(self.mode_list.index(m))
            else:
                mode_ids.append(len(self.mode_list)) # Unknown
        mode_emb = self.mode_embedding(torch.tensor(mode_ids, device=device))
        
        # Numerical embedding
        num_input = torch.cat([tempos, chromaticities], dim=-1)
        num_emb = self.numerical_projection(num_input)
        
        # Concatenate and Fuse
        combined = torch.cat([text_out, mode_emb, num_emb], dim=-1)
        global_context = self.fusion(combined)
        
        return global_context
