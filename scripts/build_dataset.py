import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

import json
import yaml
from tqdm import tqdm
from tokenizer.midi_tokenizer import MIDITokenizer
from utils.midi_analysis import analyze_midi, is_valid_for_baseline
from prompt.prompt_generator import PromptGenerator
import pretty_midi

def build_dataset():
    # 1. Load config
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize components
    tokenizer = MIDITokenizer(config)
    prompt_gen = PromptGenerator()
    
    raw_dir = "data/raw_midi"
    output_dir = "data/datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_path = os.path.join(output_dir, "baseline.jsonl")
    
    # 3. Find MIDI files (recursive)
    midi_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, f))
    
    print(f"Found {len(midi_files)} MIDI files in {raw_dir}")

    count = 0
    with open(dataset_path, "w") as out_f:
        for midi_path in tqdm(midi_files, desc="Processing MIDI"):
            filename = os.path.basename(midi_path)
            folder_path = os.path.dirname(midi_path)
            
            # Detect POP909: folder name is usually a 3-digit song ID
            is_pop909 = os.path.basename(folder_path).isdigit() and len(os.path.basename(folder_path)) == 3
            
            # Skip invalid files (no piano or drums-only)
            if not is_valid_for_baseline(midi_path):
                continue
                
            try:
                # Analyze musical attributes (pass folder if POP909)
                analysis_folder = folder_path if is_pop909 else None
                attributes = analyze_midi(midi_path, pop909_folder=analysis_folder)
                
                # Generate synthetic prompt
                prompt = prompt_gen.generate(attributes)
                
                # Tokenize (POP909 Piano is Track 2, baseline assumes best track)
                if is_pop909:
                    instr_idx = 2
                else:
                    pm_file = pretty_midi.PrettyMIDI(midi_path)
                    from utils.midi_analysis import find_best_instrument
                    instr_idx = find_best_instrument(pm_file)
                    
                tokens = tokenizer.encode(midi_path, instrument_idx=instr_idx)
                
                # Save sample
                sample = {
                    "filename": filename,
                    "prompt": prompt,
                    "tokens": tokens,
                    "attributes": attributes,
                    "is_pop909": is_pop909
                }
                out_f.write(json.dumps(sample) + "\n")
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Successfully processed {count} files. Dataset saved to {dataset_path}")

if __name__ == "__main__":
    build_dataset()
