import os
import sys
import json
import tqdm

def migrate(input_path: str, output_dir: str, shard_size: int = 1000):
    """Splits a large JSONL file into smaller shards."""
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Migrating {input_path} to shards in {output_dir}...")
    
    shard_count = 1
    line_count = 0
    current_shard_file = None
    
    # We use a context manager to ensure files are closed correctly
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            # We don't know the total number of lines, so we estimate or just show progress
            # For 2.6GB, it's safer to just iterate
            for line in tqdm.tqdm(infile, desc="Sharding"):
                if line_count % shard_size == 0:
                    if current_shard_file:
                        current_shard_file.close()
                    
                    shard_name = f"shard_{shard_count:04d}.jsonl"
                    current_shard_file = open(os.path.join(output_dir, shard_name), 'w', encoding='utf-8')
                    shard_count += 1
                
                current_shard_file.write(line)
                line_count += 1
                
        if current_shard_file:
            current_shard_file.close()
            
        print(f"\n✅ Migration complete!")
        print(f"Total lines processed: {line_count}")
        print(f"Total shards created: {shard_count - 1}")
        print(f"Stored in: {output_dir}")
        print("\n[IMPORTANT] You can now safely delete the original large file.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if current_shard_file:
            current_shard_file.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_to_shards.py <input_jsonl_file>")
        sys.exit(1)
        
    INPUT_FILE = sys.argv[1]
    OUTPUT_FOLDER = "data/datasets/v2_master"
    migrate(INPUT_FILE, OUTPUT_FOLDER, shard_size=1000)
