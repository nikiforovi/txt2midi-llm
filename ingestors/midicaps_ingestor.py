import os
from datasets import load_dataset
from ingestors.base_ingestor import BaseIngestor
from typing import List, Dict, Any
import tqdm

class MidiCapsIngestor(BaseIngestor):
    """Ingestor for the MidiCaps dataset from Hugging Face."""
    
    def __init__(self, output_dir: str = "data/raw_midi/midicaps"):
        super().__init__(output_dir)
        self.dataset_name = "amaai-lab/midicaps"

    def ingest(self, limit: int = None) -> List[Dict[str, Any]]:
        """Downloads midicaps and processes files."""
        print(f"Loading {self.dataset_name} from Hugging Face...")
        
        # Load metadata
        ds = load_dataset(self.dataset_name, split='train', streaming=True)
        
        processed_data = []
        count = 0
        
        # In a real environment, we'd need to download the actual .mid files.
        # MidiCaps dataset usually points to filenames that exist in Lakh or are provided in the repo.
        # For this implementation, we assume the user has the files or we mock the download path.
        
        for item in ds:
            if limit and count >= limit:
                break
                
            # item contains: 'location', 'caption', 'tempo', 'genre', etc.
            midi_filename = item['location']
            caption = item['caption']
            
            # Placeholder: In a real run, we would locate the file on disk
            # For MidiCaps, the files are often in a 'zip' or separate repo.
            # We'll use a search path logic.
            midi_path = self._find_midi_file(midi_filename)
            
            if midi_path and os.path.exists(midi_path):
                entry = self.standardize_entry(midi_path, {
                    "source": "midicaps",
                    "original_caption": caption,
                    "genre": item.get("genre", "unknown"),
                    "mood": item.get("mood", "unknown")
                })
                
                if entry:
                    # Enrich prompt with Midicaps caption
                    entry["prompt"] = f"{caption} (Tempo: {entry['tempo']}, Scale: {entry['scale']})"
                    processed_data.append(entry)
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} MidiCaps files...")
            
        return processed_data

    def _find_midi_file(self, filename: str) -> str:
        """Heuristic to find the MIDI file locally."""
        # Check in output_dir and common subfolders
        search_paths = [
            self.output_dir,
            os.path.join(self.output_dir, "midis"),
            "data/raw_midi/lmd_full" # Often MidiCaps points to Lakh filenames
        ]
        
        for p in search_paths:
            full_path = os.path.join(p, filename)
            if os.path.exists(full_path):
                return full_path
        return None
