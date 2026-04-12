import torch
from torch.utils.data import Dataset
import os
import json
import glob
import tqdm
from typing import List, Dict, Any
from tokenizer.midi_tokenizer import MIDITokenizer

class MIDIDatasetV2(Dataset):
    """Dataset for v2 Text-to-MIDI training with support for sharded JSONL files."""
    
    def __init__(self, data_dir: str, tokenizer: MIDITokenizer, max_len: int = 1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        """Loads samples from a single file or a directory of shards."""
        # 1. Check for sharded directory
        shard_dir = os.path.join(self.data_dir, "v2_master")
        if os.path.isdir(shard_dir):
            shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.jsonl")))
            if not shard_files:
                print(f"Warning: No .jsonl files found in {shard_dir}")
            else:
                print(f"Found {len(shard_files)} shards. Loading...")
                for sf in tqdm.tqdm(shard_files, desc="Loading shards"):
                    with open(sf, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                self.samples.append(json.loads(line))
                print(f"Successfully loaded {len(self.samples)} samples.")
                return

        # 2. Fallback to single file
        dataset_path = os.path.join(self.data_dir, "v2_dataset.jsonl")
        if os.path.exists(dataset_path):
            print(f"Loading single dataset file {dataset_path}...")
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
            print(f"Successfully loaded {len(self.samples)} samples.")
        else:
            print(f"Warning: No dataset found in {self.data_dir}. Please run migration or ingestion.")

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
