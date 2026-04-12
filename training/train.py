import torch
import os
import sys
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.getcwd())

from model.music_transformer import MusicTransformer
from model.prompt_encoder import PromptEncoderV2
from tokenizer.midi_tokenizer import MIDITokenizer
from training.dataset import MIDIDatasetV2

def train_v2():
    # 1. Load Config
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v2 Training on {device}")

    # 2. Initialize Tokenizer & Model
    tokenizer = MIDITokenizer(config)
    
    # Update vocab_size in config from tokenizer
    config['model']['vocab_size'] = tokenizer.vocab_size
    
    prompt_encoder = PromptEncoderV2(embedding_dim=config['model']['hidden_size']).to(device)
    music_model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['hidden_size'],
        nhead=config['model']['heads'],
        num_layers=config['model']['layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        max_seq_len=config['model']['context_length']
    ).to(device)

    # 3. Setup Dataset
    dataset = MIDIDatasetV2("data/datasets", tokenizer, max_len=config['model']['context_length'])
    if len(dataset) == 0:
        print("Empty dataset. Please run scripts/build_dataset.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 4. Optimizer and Loss
    # We optimize MusicTransformer AND the learnable parts of PromptEncoderV2 (mode embeddings, MLP, fusion)
    optimizer = torch.optim.Adam(
        list(music_model.parameters()) + list(prompt_encoder.mode_embedding.parameters()) + 
        list(prompt_encoder.numerical_projection.parameters()) + list(prompt_encoder.fusion.parameters()),
        lr=config['training']['learning_rate']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is Pad_None

    # 5. Training Loop
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        music_model.train()
        prompt_encoder.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            tokens = batch["tokens"].to(device)
            prompts = batch["prompt"]
            modes = batch["mode"]
            tempos = batch["tempo"].to(device)
            chromaticities = batch["chromaticity"].to(device)
            
            optimizer.zero_grad()
            
            # Multi-factor Conditioning
            global_context = prompt_encoder(prompts, modes, tempos, chromaticities)
            
            # Forward pass: Causal language modeling
            # logits: (batch, seq_len-1, vocab_size)
            logits = music_model(tokens[:, :-1], global_context)
            
            # Loss calculation
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tokens[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
        
        # Save checkpoints
        if (epoch + 1) % config['training']['save_interval'] == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'music_model': music_model.state_dict(),
                'prompt_encoder': prompt_encoder.state_dict(),
                'config': config,
                'vocab_size': tokenizer.vocab_size
            }, f"checkpoints/v2_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_v2()
