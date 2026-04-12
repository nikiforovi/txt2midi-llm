import os
import json
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from ingestors.midicaps_ingestor import MidiCapsIngestor

def process_all_datasets(output_dir: str, limit_per_source: int = 1000, num_workers: int = None, shard_size: int = 1000):
    """Orchestrates multiple ingestors and saves results in sharded JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    
    ingestors = [
        MidiCapsIngestor()
    ]
    
    seen_hashes = set()
    master_dataset = []
    
    for ingestor in ingestors:
        print(f"--- Running Ingestor (Parallel): {ingestor.__class__.__name__} ---")
        entries = ingestor.ingest(limit=limit_per_source, num_workers=num_workers)
        
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

    # Write master dataset in shards
    print(f"Saving {len(master_dataset)} unique samples to shards in {output_dir}...")
    
    shard_count = 1
    for i in range(0, len(master_dataset), shard_size):
        shard_data = master_dataset[i:i + shard_size]
        shard_name = f"shard_{shard_count:04d}.jsonl"
        shard_path = os.path.join(output_dir, shard_name)
        
        with open(shard_path, "w", encoding="utf-8") as f:
            for entry in shard_data:
                f.write(json.dumps(entry) + "\n")
        
        shard_count += 1
            
    print(f"\n✅ All datasets processed! Shards saved to {output_dir}")

if __name__ == "__main__":
    # Now pointing to a directory instead of a file
    MASTER_DIR = "data/datasets/v2_master"
    LIMIT = 50000 
    WORKERS = os.cpu_count() 
    
    process_all_datasets(MASTER_DIR, limit_per_source=LIMIT, num_workers=WORKERS, shard_size=1000)
