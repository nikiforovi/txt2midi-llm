import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument
from typing import List, Dict, Any, Tuple

class MIDITokenizer:
    """REMI+ Tokenizer for Multi-Instrument MIDI files."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.beat_res = 4  # 4 slots per beat (16th notes)
        self.ticks_per_beat = 480
        self.position_res = self.ticks_per_beat // self.beat_res
        
        self.vocab = []
        self._build_vocab()
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def _build_vocab(self):
        """Builds expanded REMI+ vocabulary."""
        self.vocab.append("Pad_None")
        self.vocab.append("Bar_None")
        
        # 1. Position (0-15)
        for i in range(16):
            self.vocab.append(f"Position_{i}")
            
        # 2. Instrument (General MIDI 0-127)
        for i in range(128):
            self.vocab.append(f"Instrument_{i}")
            
        # 3. Note_On (Pitch 0-127)
        for i in range(128):
            self.vocab.append(f"Note_Pitch_{i}")
            
        # 4. Note_Duration (0-32 16th notes)
        for i in range(1, 33):
            self.vocab.append(f"Note_Duration_{i}")
            
        # 5. Note_Velocity (0-127 quantized to 8 levels)
        for i in range(0, 128, 16):
            self.vocab.append(f"Note_Velocity_{i}")

    def encode(self, midi_path: str) -> List[int]:
        """Encodes multi-track MIDI into a interleaved REMI+ stream."""
        midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
        ticks_per_beat = midi_obj.ticks_per_beat
        pos_res = ticks_per_beat // self.beat_res
        
        all_notes = []
        for inst in midi_obj.instruments:
            if inst.is_drum: continue # Drums handled separately or ignored for now
            for note in inst.notes:
                all_notes.append({
                    "pitch": note.pitch,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity,
                    "program": inst.program
                })
        
        # Sort by start time, then pitch
        all_notes.sort(key=lambda x: (x["start"], x["pitch"]))
        
        tokens = []
        last_bar = -1
        last_pos = -1
        last_program = -1
        
        for note in all_notes:
            bar = note["start"] // (ticks_per_beat * 4)
            pos = (note["start"] % (ticks_per_beat * 4)) // pos_res
            
            # 1. Time Context
            if bar > last_bar:
                for _ in range(bar - last_bar):
                    tokens.append("Bar_None")
                last_bar = bar
                last_pos = -1 # Reset pos on new bar
            
            if pos > last_pos:
                tokens.append(f"Position_{int(pos)}")
                last_pos = pos
            
            # 2. Instrument Context
            if note["program"] != last_program:
                tokens.append(f"Instrument_{note['program']}")
                last_program = note["program"]
            
            # 3. Note Data
            tokens.append(f"Note_Pitch_{note['pitch']}")
            
            dur = max(1, round((note["end"] - note["start"]) / pos_res))
            dur = min(dur, 32)
            tokens.append(f"Note_Duration_{dur}")
            
            vel = (note["velocity"] // 16) * 16
            tokens.append(f"Note_Velocity_{vel}")
            
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, token_ids: List[int], output_path: str, target_bpm: float = None):
        """Decodes REMI+ tokens back to multi-track MIDI."""
        tokens = [self.id_to_token[tid] for tid in token_ids]
        
        midi_obj = miditoolkit.midi.parser.MidiFile()
        midi_obj.ticks_per_beat = self.ticks_per_beat
        pos_res = self.ticks_per_beat // self.beat_res
        
        # If target_bpm is requested, we adjust ticks accordingly
        # But usually we just set the tempo meta-message
        if target_bpm:
            # We add a tempo change at tick 0
            midi_obj.tempo_changes.append(miditoolkit.TempoChange(target_bpm, 0))

        instruments = {} # Store instrument objects by program
        
        current_bar = -1
        current_pos = 0
        current_program = 0 # Default Piano
        
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "Bar_None":
                current_bar += 1
                i += 1
            elif t.startswith("Position_"):
                current_pos = int(t.split("_")[1])
                i += 1
            elif t.startswith("Instrument_"):
                current_program = int(t.split("_")[1])
                i += 1
            elif t.startswith("Note_Pitch_"):
                pitch = int(t.split("_")[2])
                dur = int(tokens[i+1].split("_")[2])
                vel = int(tokens[i+2].split("_")[2])
                
                if current_program not in instruments:
                    instruments[current_program] = Instrument(program=current_program, is_drum=False)
                
                start_tick = current_bar * (self.ticks_per_beat * 4) + current_pos * pos_res
                end_tick = start_tick + dur * pos_res
                
                instruments[current_program].notes.append(Note(
                    pitch=pitch, velocity=vel, start=start_tick, end=end_tick
                ))
                i += 3
            else:
                i += 1
                
        for inst in instruments.values():
            midi_obj.instruments.append(inst)
            
        midi_obj.dump(output_path)
