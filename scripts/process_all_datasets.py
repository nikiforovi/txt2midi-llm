import os
import json
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from ingestors.midicaps_ingestor import MidiCapsIngestor
# from ingestors.lmd_ingestor import LakhIngestor # To be implemented

def process_all_datasets(output_master_path: str, limit_per_source: int = 1000):
    """Orchestrates multiple ingestors and performs global deduplication."""
    os.makedirs(os.path.dirname(output_master_path), exist_ok=True)
    
    ingestors = [
        MidiCapsIngestor()
    ]
    
    seen_hashes = set()
    master_dataset = []
    
    for ingestor in ingestors:
        print(f"--- Running Ingestor: {ingestor.__class__.__name__} ---")
        entries = ingestor.ingest(limit=limit_per_source)
        
        raw_count = len(entries)
        duplicates = 0
        
        for entry in entries:
            h = entry["hash"]
            if h not in seen_hashes:
                seen_hashes.add(h)
                master_dataset.append(entry)
            else:
                duplicates += 1
                
        print(f"Ingested {raw_count} samples. Unique: {raw_count - duplicates}. Duplicates: {duplicates}")

    # Write master dataset
    with open(output_master_path, "w") as f:
        for entry in master_dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\n✅ All datasets processed! Master dataset saved to {output_master_path}")
    print(f"Total unique samples: {len(master_dataset)}")

if __name__ == "__main__":
    # For a start, let's ingest a small batch to verify everything works
    MASTER_PATH = "data/datasets/v2_master_dataset.jsonl"
    process_all_datasets(MASTER_PATH, limit_per_source=5000)
