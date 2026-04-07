import pretty_midi
import os
import numpy as np
from typing import Dict, Any, Tuple

def parse_pop909_annotations(folder_path: str) -> Dict[str, Any]:
    """Parses POP909 text annotations (key_audio.txt, key.txt)."""
    attr = {}
    
    # Check possible filenames
    for fname in ["key_audio.txt", "key_midi.txt", "key.txt"]:
        key_path = os.path.join(folder_path, fname)
        if os.path.exists(key_path):
            try:
                with open(key_path, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    if not lines: continue
                    
                    # Take the first line (baseline: assume it's the main key)
                    # Format: start_time \t end_time \t key
                    parts = lines[0].split()
                    if len(parts) >= 3:
                        key_str = parts[2] # "Gb:maj" or "Am"
                        if ":" in key_str:
                            k, s = key_str.split(":")
                            attr["key"] = k
                            attr["scale"] = "major" if s.startswith("maj") else "minor"
                        elif key_str.endswith("m") and len(key_str) > 1:
                             attr["key"] = key_str[:-1]
                             attr["scale"] = "minor"
                        else:
                            attr["key"] = key_str
                            attr["scale"] = "major"
                    break
            except Exception as e:
                print(f"Warning: Failed to parse {key_path}: {e}")
    
    return attr

def detect_key_heuristic(pm: pretty_midi.PrettyMIDI) -> Tuple[str, str]:
    """Detects key using pitch class histogram correlation (Krumhansl-Schmuckler)."""
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    histogram = pm.get_pitch_class_histogram()
    if np.sum(histogram) == 0:
        return "C", "major"

    best_corr = -1.1
    best_key = "C"
    best_scale = "major"

    for i in range(12):
        shifted_major = np.roll(MAJOR_PROFILE, i)
        shifted_minor = np.roll(MINOR_PROFILE, i)

        corr_major = np.corrcoef(histogram, shifted_major)[0, 1]
        corr_minor = np.corrcoef(histogram, shifted_minor)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = KEYS[i]
            best_scale = "major"
        
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = KEYS[i]
            best_scale = "minor"

    return best_key, best_scale

def find_best_instrument(pm: pretty_midi.PrettyMIDI) -> int:
    """Finds the most suitable instrument track (usually the one with most notes)."""
    if not pm.instruments:
        return 0
    best_idx = 0
    max_notes = -1
    for i, inst in enumerate(pm.instruments):
        if inst.is_drum: continue
        num_notes = len(inst.notes)
        if num_notes > max_notes:
            max_notes = num_notes
            best_idx = i
    return best_idx

def analyze_midi(midi_path: str, pop909_folder: str = None) -> Dict[str, Any]:
    """Analyzes a MIDI file and extracts musical attributes."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        # Fallback for broken MIDI
        return {"tempo": 120, "tempo_label": "moderate", "key": "C", "scale": "major", "density": "medium", "genre": "pop"}
    
    # 1. Base Analysis (Tempo)
    tempo = 120.0
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[1]) > 0:
        tempo = tempo_changes[1][0]
    
    # 2. Key Detection (Tiered)
    key_name = None
    scale_type = None

    if pop909_folder:
        pop_attr = parse_pop909_annotations(pop909_folder)
        key_name = pop_attr.get("key")
        scale_type = pop_attr.get("scale")

    if not key_name and len(pm.key_signature_changes) > 0:
        try:
            key_sig = pm.key_signature_changes[0]
            key_name_full = pretty_midi.key_number_to_key_name(key_sig.key_number)
            parts = key_name_full.split()
            key_name = parts[0]
            scale_type = parts[1].lower() if len(parts) > 1 else "major"
        except:
            pass

    if not key_name:
        key_name, scale_type = detect_key_heuristic(pm)

    # 3. Density and Labels
    total_notes = sum(len(i.notes) for i in pm.instruments if not i.is_drum)
    duration = pm.get_end_time()
    density = total_notes / duration if duration > 0 else 0
    
    density_label = "medium"
    if density < 2: density_label = "low"
    elif density > 6: density_label = "high"

    tempo_label = "moderate"
    if tempo < 90: tempo_label = "slow"
    elif tempo > 130: tempo_label = "fast"

    return {
        "tempo": round(tempo),
        "tempo_label": tempo_label,
        "key": key_name,
        "scale": scale_type,
        "density": density_label,
        "genre": "pop"
    }

def is_valid_for_baseline(midi_path: str) -> bool:
    """Checks if MIDI is suitable for the single-track baseline."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        for instrument in pm.instruments:
            if not instrument.is_drum and len(instrument.notes) > 10:
                return True
        return False
    except:
        return False
