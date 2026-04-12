import torch
from torch.utils.data import Dataset
import os
import json
from typing import List, Dict, Any
from tokenizer.midi_tokenizer import MIDITokenizer

class MIDIDatasetV2(Dataset):
    """Dataset for v2 Text-to-MIDI training with granular conditioning."""
    
    def __init__(self, data_dir: str, tokenizer: MIDITokenizer, max_len: int = 1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        dataset_path = os.path.join(self.data_dir, "v2_dataset.jsonl")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            print(f"Warning: Dataset file {dataset_path} not found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Conditioning factors
        prompt = sample.get("prompt", "")
        mode = sample.get("scale", "major")
        tempo = float(sample.get("tempo", 120)) / 300.0 # Normalize BPM
        chromaticity = float(sample.get("chromaticity", 0.0))
        
        # 2. Tokenization handling
        tokens = sample["tokens"]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            
        token_tensor = torch.zeros(self.max_len, dtype=torch.long)
        token_tensor[:len(tokens)] = torch.tensor(tokens)
        
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:len(tokens)] = True
        
        return {
            "prompt": prompt,
            "mode": mode,
            "tempo": torch.tensor([tempo], dtype=torch.float32),
            "chromaticity": torch.tensor([chromaticity], dtype=torch.float32),
            "tokens": token_tensor,
            "mask": mask,
            "seq_len": len(tokens)
        }
