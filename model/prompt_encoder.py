import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class PromptEncoder(nn.Module):
    """Encodes text prompts into embeddings using a frozen small Transformer."""
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny", embedding_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Project BERT hidden state to model hidden state
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        
        # Freeze BERT for baseline to speed up training
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, prompts: list[str]):
        """Encodes a list of text prompts.
        
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        projected = self.projection(cls_embedding)
        return projected
