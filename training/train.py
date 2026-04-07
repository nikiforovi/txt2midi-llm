import torch
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from model.music_transformer import MusicTransformer
from model.prompt_encoder import PromptEncoder
from tokenizer.midi_tokenizer import MIDITokenizer
from training.dataset import MIDIDataset

def train():
    # 1. Load Config
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 2. Initialize Tokenizer & Model
    tokenizer = MIDITokenizer(config)
    
    prompt_encoder = PromptEncoder(embedding_dim=config['model']['hidden_size']).to(device)
    music_model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['hidden_size'],
        nhead=config['model']['heads'],
        num_layers=config['model']['layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        max_seq_len=config['model']['context_length']
    ).to(device)

    # 3. Setup Dataset
    dataset = MIDIDataset("data/datasets", tokenizer, max_len=config['model']['context_length'])
    if len(dataset) == 0:
        print("Empty dataset. Please populate data/datasets/baseline.jsonl")
        # return # Commented out for baseline script visibility

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 4. Optimizer and Loss
    optimizer = torch.optim.Adam(
        list(music_model.parameters()) + list(prompt_encoder.projection.parameters()),
        lr=config['training']['learning_rate']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is padding (Pad_None)

    # 5. Training Loop
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        music_model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            prompts = batch["prompt"]
            tokens = batch["tokens"].to(device)
            
            optimizer.zero_grad()
            
            # Encode prompt
            prompt_embeds = prompt_encoder(prompts)
            
            # Forward pass
            # For causal modeling, input is tokens[:, :-1], target is tokens[:, 1:]
            logits = music_model(tokens[:, :-1], prompt_embeds)
            
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tokens[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader) if len(dataloader) > 0 else 0}")
        
        # Save checkpoints
        if (epoch + 1) % config['training']['save_interval'] == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'music_model': music_model.state_dict(),
                'prompt_encoder': prompt_encoder.state_dict(),
                'config': config
            }, f"checkpoints/baseline_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
