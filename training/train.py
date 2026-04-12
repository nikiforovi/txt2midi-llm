import torch
import os
import sys
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to sys.path
sys.path.append(os.getcwd())

from model.music_transformer import MusicTransformer
from model.prompt_encoder import PromptEncoderV2
from tokenizer.midi_tokenizer import MIDITokenizer
from training.dataset import MIDIDatasetV2

def plot_loss(losses: list, output_path: str):
    """Simple matplotlib plotter for training monitoring."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('txt2midi v2 Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def train_v2():
    # 1. Load Config
    config_path = "configs/model_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- txt2midi v2 Training Launch ---")
    print(f"Device: {device}")
    print(f"Parameters: Hidden={config['model']['hidden_size']}, Layers={config['model']['layers']}")

    # 2. Initialize Tokenizer & Models
    tokenizer = MIDITokenizer(config)
    
    print("Initializing PromptEncoderV2 (loading BERT weights)...")
    prompt_encoder = PromptEncoderV2(embedding_dim=config['model']['hidden_size']).to(device)
    
    print("Building MusicTransformer architecture...")
    music_model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['hidden_size'],
        nhead=config['model']['heads'],
        num_layers=config['model']['layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        max_seq_len=config['model']['context_length']
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"--- DETECTED {torch.cuda.device_count()} GPUs! Enabling DataParallel ---")
        music_model = nn.DataParallel(music_model)

    # 3. Setup Dataset (Sharded support)
    dataset = MIDIDatasetV2("data/datasets", tokenizer, max_len=config['model']['context_length'])
    if len(dataset) == 0:
        print("Dataset is empty. Ensure data/datasets/v2_master contains shard_*.jsonl files.")
        return

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)

    # 4. Optimizer and Scheduler
    # We optimize EVERYTHING (Transformer + PromptEncoder conditioning layers)
    optimizer = torch.optim.AdamW(
        list(music_model.parameters()) + list(prompt_encoder.parameters()),
        lr=config['training']['learning_rate']
    )
    
    # Cosine Annealing: high LR at start, gradually decreasing to 1/10th or less
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-5)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is Pad_None

    # 5. Training Loop
    epochs = config['training']['epochs']
    epoch_losses = []
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        music_model.train()
        prompt_encoder.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            tokens = batch["tokens"].to(device)
            prompts = batch["prompt"]
            modes = batch["mode"]
            tempos = batch["tempo"].to(device)
            chromaticities = batch["chromaticity"].to(device)
            
            optimizer.zero_grad()
            
            # Multi-factor Conditioning
            global_context = prompt_encoder(prompts, modes, tempos, chromaticities)
            
            # Forward pass
            logits = music_model(tokens[:, :-1], global_context)
            
            # Loss: Target is tokens shifted by 1
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tokens[:, 1:].reshape(-1))
            
            loss.backward()
            
            # Gradient clipping to prevent spikes
            # If multi-GPU, music_model might be DataParallel, but parameters() still works
            torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Update plot
        plot_loss(epoch_losses, "checkpoints/loss_curve.png")
        
        # Save checkpoints
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = f"checkpoints/v2_epoch_{epoch+1}.pth"
            
            # Handle DataParallel state dict (remove 'module.' prefix)
            model_to_save = music_model.module if hasattr(music_model, 'module') else music_model
            
            torch.save({
                'epoch': epoch + 1,
                'music_model': model_to_save.state_dict(),
                'prompt_encoder': prompt_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'losses': epoch_losses
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    train_v2()
