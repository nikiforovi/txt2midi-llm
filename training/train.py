import torch
import os
import sys
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

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
    plt.title('txt2midi v2 Turbo Training Progress')
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

    # Detect Kaggle environment for checkpointing
    is_kaggle = os.path.exists('/kaggle/working')
    output_dir = '/kaggle/working/outputs_v2' if is_kaggle else 'outputs_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Path for the loss curve - keep it in root for Kaggle to be easily visible
    plot_path = '/kaggle/working/loss_curve.png' if is_kaggle else os.path.join(output_dir, "loss_curve.png")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- txt2midi v2 TURBO Training Launch ---")
    print(f"Device: {device} | Total GPUs: {torch.cuda.device_count()}")
    print(f"AMP: Enabled | Multi-threading: 4 workers")

    # 2. Initialize Tokenizer & Models
    tokenizer = MIDITokenizer(config)
    
    print("Initializing PromptEncoderV2 (CPU BERT)...")
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
        print(f"--- Enabling DataParallel for {torch.cuda.device_count()} GPUs ---")
        music_model = nn.DataParallel(music_model)

    # 3. Setup Dataset
    dataset = MIDIDatasetV2("data/datasets", tokenizer, max_len=config['model']['context_length'])
    if len(dataset) == 0:
        print("Dataset is empty. Ensure shards are present.")
        return

    # Turbo loading: 4 workers + pin_memory
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    # 4. Optimizer and Speed-ups
    optimizer = torch.optim.AdamW(
        list(music_model.parameters()) + list(prompt_encoder.parameters()),
        lr=config['training']['learning_rate']
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-5)
    scaler = GradScaler() # For Mixed Precision
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    # 5. Training Loop
    epochs = config['training']['epochs']
    epoch_losses = []
    
    for epoch in range(epochs):
        music_model.train()
        prompt_encoder.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            tokens = batch["tokens"].to(device, non_blocking=True)
            prompts = batch["prompt"]
            modes = batch["mode"]
            tempos = batch["tempo"].to(device, non_blocking=True)
            chromaticities = batch["chromaticity"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use Mixed Precision (AMP)
            with autocast():
                # Conditioning
                global_context = prompt_encoder(prompts, modes, tempos, chromaticities)
                
                # Forward pass
                logits = music_model(tokens[:, :-1], global_context)
                
                # Loss calculation
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tokens[:, 1:].reshape(-1))
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale before clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_norm=1.0)
            
            # Scaled step
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Update plot
        plot_loss(epoch_losses, plot_path)
        
        # Save checkpoints
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(output_dir, f"v2_turbo_epoch_{epoch+1}.pth")
            
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

    print("\n✅ TURBO Training Complete!")

if __name__ == "__main__":
    train_v2()
