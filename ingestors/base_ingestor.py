import os
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import sys
from concurrent.futures import ProcessPoolExecutor
import tqdm

# Add project root to sys.path
sys.path.append(os.getcwd())

from utils.midi_analysis import analyze_midi_v2, is_valid_v2
from tokenizer.midi_tokenizer import MIDITokenizer

def _worker_process(args: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Independent worker function for parallel processing."""
    midi_path, raw_metadata = args
    
    if not is_valid_v2(midi_path):
        return None
        
    try:
        # We re-initialize the tokenizer in each worker to avoid pickling issues
        # although with small vocab it's cheap.
        config = {"model": {"vocab_size": 512}}
        tokenizer = MIDITokenizer(config)
        
        analysis = analyze_midi_v2(midi_path)
        tokens = tokenizer.encode(midi_path)
        
        if len(tokens) < 10:
            return None
            
        # Get content hash
        hasher = hashlib.md5()
        with open(midi_path, 'rb') as f:
            hasher.update(f.read())
        content_hash = hasher.hexdigest()
            
        # Combine and ensure JSON serializable types
        entry = {
            "hash": content_hash,
            "original_path": midi_path,
            "tokens": [int(t) for t in tokens],
            "tempo": int(analysis["tempo"]),
            "key": str(analysis["key"]),
            "scale": str(analysis["scale"]),
            "chromaticity": float(analysis["chromaticity"]),
            "instruments": [int(i) for i in analysis["instruments"]],
            "metadata": raw_metadata 
        }
        return entry
    except Exception:
        return None

class BaseIngestor(ABC):
    """Abstract base class for dataset-specific ingestors with multiprocessing support."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    @abstractmethod
    def ingest(self, limit: int = None, num_workers: int = None) -> List[Dict[str, Any]]:
        """Processes the dataset and returns a list of standardized entries."""
        pass

    def run_parallel(self, tasks: List[Tuple[str, Dict[str, Any]]], num_workers: int = None) -> List[Dict[str, Any]]:
        """Executes processing tasks in parallel using a process pool."""
        if num_workers is None:
            num_workers = os.cpu_count()
            
        print(f"Starting parallel ingestion with {num_workers} workers...")
        
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Using list() to consume the map and show progress
            for result in tqdm.tqdm(executor.map(_worker_process, tasks), total=len(tasks), desc="Parallel Processing"):
                if result:
                    results.append(result)
                    
        return results
