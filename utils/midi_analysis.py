import pretty_midi
import os
import numpy as np
from typing import Dict, Any, Tuple, List

# Musical Mode Profiles (based on Krumhansl-Schmuckler and extensions)
MODAL_PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "dorian": [6.08, 2.14, 3.33, 2.23, 4.19, 3.91, 2.41, 4.96, 2.29, 3.50, 2.19, 2.76],
    "phrygian": [6.08, 4.19, 3.33, 2.23, 3.91, 2.41, 4.96, 2.29, 3.50, 2.19, 2.76, 2.14],
    "lydian": [6.35, 2.23, 3.48, 2.33, 4.38, 2.52, 4.09, 5.19, 2.39, 3.66, 2.29, 2.88],
    "mixolydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.88, 2.29],
    "locrian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 4.75, 2.54, 3.98, 2.69, 3.34, 3.17],
    # Exotic/Jazz Modes
    "dorian_b2": [6.08, 4.19, 3.33, 2.23, 4.19, 3.91, 2.41, 4.96, 2.29, 3.50, 2.19, 2.76], # Phrygian #6
    "lydian_dominant": [6.35, 2.23, 3.48, 2.33, 4.38, 2.52, 4.09, 5.19, 2.88, 3.66, 2.29, 2.88],
}

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def parse_pop909_annotations(folder_path: str) -> Dict[str, Any]:
    """Parses POP909 text annotations with high precision."""
    attr = {}
    for fname in ["key_audio.txt", "key_midi.txt", "key.txt"]:
        key_path = os.path.join(folder_path, fname)
        if os.path.exists(key_path):
            try:
                with open(key_path, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    if not lines: continue
                    parts = lines[0].split()
                    if len(parts) >= 3:
                        key_str = parts[2]
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
            except: pass
    return attr

def detect_mode_heuristic(pm: pretty_midi.PrettyMIDI) -> Tuple[str, str]:
    """Detects mode using correlation with multiple profiles."""
    histogram = pm.get_pitch_class_histogram()
    if np.sum(histogram) == 0:
        return "C", "major"

    best_score = -2.0
    best_key = "C"
    best_mode = "major"

    for mode_name, profile in MODAL_PROFILES.items():
        profile_np = np.array(profile)
        for i in range(12):
            shifted_profile = np.roll(profile_np, i)
            score = np.corrcoef(histogram, shifted_profile)[0, 1]
            if score > best_score:
                best_score = score
                best_key = KEYS[i]
                best_mode = mode_name
                
    return best_key, best_mode

def calculate_chromaticity(pm: pretty_midi.PrettyMIDI, root: str, mode: str) -> float:
    """Calculates the ratio of notes outside the detected scale."""
    # Define scale degrees for each mode
    MODE_DEGREES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "locrian": [0, 1, 3, 5, 6, 8, 10],
        "dorian_b2": [0, 1, 3, 5, 7, 9, 10],
        "lydian_dominant": [0, 2, 4, 6, 7, 9, 10]
    }
    
    degrees = MODE_DEGREES.get(mode, MODE_DEGREES["major"])
    root_idx = KEYS.index(root)
    scale_pitches = [(root_idx + d) % 12 for d in degrees]
    
    total_notes = 0
    out_of_scale_notes = 0
    
    for inst in pm.instruments:
        if inst.is_drum: continue
        for note in inst.notes:
            total_notes += 1
            if (note.pitch % 12) not in scale_pitches:
                out_of_scale_notes += 1
                
    return out_of_scale_notes / total_notes if total_notes > 0 else 0.0

def analyze_midi_v2(midi_path: str, pop909_folder: str = None) -> Dict[str, Any]:
    """Enhanced analysis for v2: multi-track, modal detection, chromaticity."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except:
        return {"tempo": 120, "key": "C", "scale": "major", "chromaticity": 0.0, "instruments": [0]}
    
    # 1. Multi-track Metadata
    instruments = []
    for inst in pm.instruments:
        if not inst.is_drum:
            instruments.append(inst.program)
    if not instruments: instruments = [0]
    
    # 2. Tempo
    tempo = 120.0
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[1]) > 0:
        tempo = tempo_changes[1][0]
    
    # 3. Key & Mode Detection
    key_name = None
    mode_name = None

    if pop909_folder:
        pop_attr = parse_pop909_annotations(pop909_folder)
        key_name = pop_attr.get("key")
        mode_name = pop_attr.get("scale")

    if not key_name:
        key_name, mode_name = detect_mode_heuristic(pm)

    # 4. Chromaticity
    chromaticity = calculate_chromaticity(pm, key_name, mode_name)

    return {
        "tempo": round(tempo),
        "key": key_name,
        "scale": mode_name,
        "chromaticity": round(chromaticity, 3),
        "instruments": list(set(instruments)),
        "duration": round(pm.get_end_time(), 2)
    }

def is_valid_v2(midi_path: str) -> bool:
    """Validator for v2 datasets."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        return len(pm.instruments) > 0 and pm.get_end_time() > 5
    except:
        return False
