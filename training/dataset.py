import torch
from torch.utils.data import Dataset
import os
import json
from typing import List, Dict, Any
from tokenizer.midi_tokenizer import MIDITokenizer

class MIDIDataset(Dataset):
    """Dataset for Text-to-MIDI training."""
    
    def __init__(self, data_dir: str, tokenizer: MIDITokenizer, max_len: int = 1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # We expect a JSONL file or a directory of (prompt, midi) pairs
        # For the baseline, we'll assume a list of files in data/datasets/baseline.jsonl
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        dataset_path = os.path.join(self.data_dir, "baseline.jsonl")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            # Fallback: empty dataset if not found
            print(f"Warning: Dataset file {dataset_path} not found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        tokens = sample["tokens"] # Pre-tokenized for speed or raw midi path
        
        # Ensure tokens are limited by max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            
        # Pad or truncate
        token_tensor = torch.zeros(self.max_len, dtype=torch.long)
        token_tensor[:len(tokens)] = torch.tensor(tokens)
        
        # Create mask for valid tokens (ignoring padding in loss)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:len(tokens)] = True
        
        return {
            "prompt": prompt,
            "tokens": token_tensor,
            "mask": mask,
            "seq_len": len(tokens)
        }
