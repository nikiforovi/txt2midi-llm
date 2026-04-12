import os
import json
from ingestors.base_ingestor import BaseIngestor
from typing import List, Dict, Any
import tqdm

class MidiCapsIngestor(BaseIngestor):
    """Ingestor for the MidiCaps dataset using local files with parallel processing."""
    
    def __init__(self, base_dir: str = "data/raw_midi/midicaps"):
        super().__init__(base_dir)
        self.base_dir = base_dir
        self.metadata_path = os.path.join(base_dir, "train.json")

    def ingest(self, limit: int = None, num_workers: int = None) -> List[Dict[str, Any]]:
        """Collects tasks from JSONL and processes them in parallel."""
        if not os.path.exists(self.metadata_path):
            print(f"Error: Metadata file {self.metadata_path} not found.")
            return []
            
        tasks = []
        count = 0
        
        print(f"Scanning metadata and collecting tasks from {self.metadata_path}...")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    item = json.loads(line)
                    midi_rel_path = item['location']
                    caption = item['caption']
                    
                    midi_path = os.path.join(self.base_dir, midi_rel_path)
                    
                    if os.path.exists(midi_path):
                        # Prepare metadata to be passed to worker
                        raw_meta = {
                            "source": "midicaps",
                            "original_caption": caption,
                            "genre": item.get("genre", []),
                            "mood": item.get("mood", "unknown")
                        }
                        tasks.append((midi_path, raw_meta))
                        count += 1
                except:
                    continue
        
        print(f"Collected {len(tasks)} valid tasks. Starting parallel processing...")
        
        # Run parallel processing via BaseIngestor method
        entries = self.run_parallel(tasks, num_workers=num_workers)
        
        # Post-process prompts (can be done parallel too, but it's cheap strings)
        for entry in entries:
            # Enrich entry with technical context if needed
             entry["prompt"] = entry["metadata"]["original_caption"]
             
        return entries
