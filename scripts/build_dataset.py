import os
import json
import tqdm
import sys
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from utils.midi_analysis import analyze_midi_v2, is_valid_v2
from tokenizer.midi_tokenizer import MIDITokenizer
from prompt.prompt_generator import PromptGenerator

def build_dataset_v2(raw_midi_dir: str, output_path: str):
    """Builds a dataset with granular v2 metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    config = {"tokenizer": {"type": "REMI+"}} # Minimal config for initializer
    tokenizer = MIDITokenizer(config)
    prompt_gen = PromptGenerator()
    
    midi_files = []
    for root, _, files in os.walk(raw_midi_dir):
        for f in files:
            if f.endswith(".mid") or f.endswith(".midi"):
                midi_files.append(os.path.join(root, f))
                
    print(f"Found {len(midi_files)} MIDI files in {raw_midi_dir}")
    
    dataset = []
    success_count = 0
    
    for midi_path in tqdm.tqdm(midi_files, desc="Processing MIDI v2"):
        if not is_valid_v2(midi_path):
            continue
            
        try:
            # 1. Advanced Analysis
            # Try to find POP909 folder if applicable
            pop_folder = None
            parent = os.path.dirname(midi_path)
            if os.path.basename(parent).isdigit() or os.path.basename(os.path.dirname(parent)).isdigit():
                pop_folder = parent if os.path.basename(parent).isdigit() else os.path.dirname(parent)

            analysis = analyze_midi_v2(midi_path, pop909_folder=pop_folder)
            
            # 2. Tokenization
            token_ids = tokenizer.encode(midi_path)
            if len(token_ids) < 10: continue
            
            # 3. Prompt Generation (can be enriched further)
            prompt = prompt_gen.generate(analysis)
            
            # 4. Save entry
            entry = {
                "id": os.path.basename(midi_path),
                "prompt": prompt,
                "tempo": analysis["tempo"],
                "key": analysis["key"],
                "scale": analysis["scale"],
                "chromaticity": analysis["chromaticity"],
                "instruments": analysis["instruments"],
                "tokens": token_ids
            }
            dataset.append(entry)
            success_count += 1
            
        except Exception as e:
            # print(f"Error processing {midi_path}: {e}")
            continue

    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully processed {success_count} files. Dataset saved to {output_path}")

if __name__ == "__main__":
    RAW_DIR = "data/raw_midi"
    OUTPUT_FILE = "data/datasets/v2_dataset.jsonl"
    build_dataset_v2(RAW_DIR, OUTPUT_FILE)
