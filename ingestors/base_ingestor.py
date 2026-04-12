import os
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from utils.midi_analysis import analyze_midi_v2, is_valid_v2
from tokenizer.midi_tokenizer import MIDITokenizer

class BaseIngestor(ABC):
    """Abstract base class for dataset-specific ingestors."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize common tools
        config = {"model": {"vocab_size": 512}}
        self.tokenizer = MIDITokenizer(config)
        
    def get_content_hash(self, midi_path: str) -> str:
        """Generates a stable hash based on the MIDI note content (ignoring meta)."""
        # For simplicity, we use MD5 of the file, but in a production system, 
        # we'd hash the note sequence to identify musical duplicates.
        hasher = hashlib.md5()
        with open(midi_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    @abstractmethod
    def ingest(self, limit: int = None) -> List[Dict[str, Any]]:
        """Processes the dataset and returns a list of standardized entries."""
        pass

    def standardize_entry(self, midi_path: str, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Runs v2 analysis and combines it with dataset metadata."""
        if not is_valid_v2(midi_path):
            return None
            
        try:
            analysis = analyze_midi_v2(midi_path)
            tokens = self.tokenizer.encode(midi_path)
            
            if len(tokens) < 10:
                return None
                
            # Combine
            entry = {
                "hash": self.get_content_hash(midi_path),
                "original_path": midi_path,
                "tokens": tokens,
                "tempo": analysis["tempo"],
                "key": analysis["key"],
                "scale": analysis["scale"],
                "chromaticity": analysis["chromaticity"],
                "instruments": analysis["instruments"],
                "metadata": raw_metadata # Source specific
            }
            return entry
        except Exception as e:
            # print(f"Error standardizing {midi_path}: {e}")
            return None
