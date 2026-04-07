import torch
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

import torch.nn.functional as F
from model.music_transformer import MusicTransformer
from model.prompt_encoder import PromptEncoder
from tokenizer.midi_tokenizer import MIDITokenizer
from prompt.prompt_generator import PromptGenerator
from typing import List

class MusicGenerator:
    """Inference class for generating MIDI from text prompts."""
    
    def __init__(self, checkpoint_path: str, config: dict, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.tokenizer = MIDITokenizer(config)
        self.prompt_gen = PromptGenerator()
        
        self.prompt_encoder = PromptEncoder(embedding_dim=config['model']['hidden_size']).to(self.device)
        self.music_model = MusicTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config['model']['hidden_size'],
            nhead=config['model']['heads'],
            num_layers=config['model']['layers']
        ).to(self.device)
        
        # Load weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.music_model.load_state_dict(checkpoint['music_model'])
            self.prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
        
        self.music_model.eval()
        self.prompt_encoder.eval()

    @torch.no_grad()
    def generate(self, text_prompt: str, max_len: int = 512, temperature: float = 1.0) -> List[int]:
        """Generates a sequence of token IDs from a text prompt."""
        
        # 1. Encode prompt
        prompt_embed = self.prompt_encoder([text_prompt])
        
        # 2. Iterative generation
        # Start with a "Bar" token or empty
        generated = [self.tokenizer.token_to_id["Bar_None"]]
        
        for _ in range(max_len):
            input_tensor = torch.tensor([generated]).to(self.device)
            
            logits = self.music_model(input_tensor, prompt_embed)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Simple heuristic to stop if too many pads (or end of song token if we had one)
            if next_token == self.tokenizer.token_to_id["Pad_None"]:
                break
                
        return generated

    def generate_to_file(self, text_prompt: str, output_path: str, **kwargs):
        """Generates MIDI tokens and saves to file."""
        token_ids = self.generate(text_prompt, **kwargs)
        self.tokenizer.decode(token_ids, output_path)
        return output_path

import os
import yaml

if __name__ == "__main__":
    # Example usage for testing structure
    with open("configs/model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Path to a dummy or real checkpoint
    gen = MusicGenerator("checkpoints/baseline_epoch_50.pth", cfg)
    print("Generator initialized. (Note: weights might be random if checkpoint missing)")
    
    # gen.generate_to_file("slow sad piano in C minor", "output.mid")
