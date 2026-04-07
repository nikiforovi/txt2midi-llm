import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument
from typing import List, Dict, Any, Tuple

class MIDITokenizer:
    """REMI Tokenizer for MIDI files."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.beat_res = 4  # 4 slots per beat for baseline (16th notes)
        self.ticks_per_beat = 480  # Default MIDI ticks per beat
        self.position_res = self.ticks_per_beat // self.beat_res
        
        # Vocab definition
        self.vocab = []
        self._build_vocab()
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def _build_vocab(self):
        """Builds the REMI vocabulary."""
        self.vocab.append("Pad_None")
        self.vocab.append("Bar_None")
        
        # Position (0-15 for 4/4 bar with 16th note resolution)
        for i in range(16):
            self.vocab.append(f"Position_{i}")
            
        # Note_On (Pitch 0-127)
        for i in range(128):
            self.vocab.append(f"Note_Pitch_{i}")
            
        # Note_Duration (0-32 16th notes)
        for i in range(1, 33):
            self.vocab.append(f"Note_Duration_{i}")
            
        # Note_Velocity (0-127 quantized to 8 levels)
        for i in range(0, 128, 16):
            self.vocab.append(f"Note_Velocity_{i}")

    def encode(self, midi_path: str, instrument_idx: int = 0) -> List[int]:
        """Encodes a MIDI file into a sequence of token IDs.
        
        Args:
            midi_path: Path to MIDI file.
            instrument_idx: Index of the instrument to encode (0 for baseline, 
                             2 for POP909 Piano Accompaniment).
        """
        midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
        ticks_per_beat = midi_obj.ticks_per_beat
        pos_res = ticks_per_beat // self.beat_res
        
        if instrument_idx >= len(midi_obj.instruments):
            instrument_idx = 0
            
        notes = midi_obj.instruments[instrument_idx].notes
        notes.sort(key=lambda x: x.start)
        
        tokens = []
        last_bar = -1
        
        for note in notes:
            # Calculate bar and position
            bar = note.start // (ticks_per_beat * 4)
            pos = (note.start % (ticks_per_beat * 4)) // pos_res
            
            # Add Bar token if needed
            if bar > last_bar:
                for _ in range(bar - last_bar):
                    tokens.append("Bar_None")
                last_bar = bar
                
            # Add Position token
            tokens.append(f"Position_{int(pos)}")
            
            # Add Note information
            tokens.append(f"Note_Pitch_{note.pitch}")
            
            # Duration (quantized to 16th notes)
            dur = max(1, round((note.end - note.start) / pos_res))
            dur = min(dur, 32)
            tokens.append(f"Note_Duration_{dur}")
            
            # Velocity (quantized)
            vel = (note.velocity // 16) * 16
            tokens.append(f"Note_Velocity_{vel}")
            
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, token_ids: List[int], output_path: str):
        """Decodes a sequence of token IDs into a MIDI file."""
        tokens = [self.id_to_token[tid] for tid in token_ids]
        
        midi_obj = miditoolkit.midi.parser.MidiFile()
        midi_obj.ticks_per_beat = self.ticks_per_beat
        pos_res = self.ticks_per_beat // self.beat_res
        
        instrument = Instrument(program=0, is_drum=False, name="Piano")
        
        current_bar = -1
        current_pos = 0
        
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "Bar_None":
                current_bar += 1
                i += 1
            elif t.startswith("Position_"):
                current_pos = int(t.split("_")[1])
                i += 1
                # Expect Pitch, Duration, Velocity follows
                if i + 2 < len(tokens) and tokens[i].startswith("Note_Pitch_"):
                    pitch = int(tokens[i].split("_")[2])
                    dur = int(tokens[i+1].split("_")[2])
                    vel = int(tokens[i+2].split("_")[2])
                    
                    start_tick = current_bar * (self.ticks_per_beat * 4) + current_pos * pos_res
                    end_tick = start_tick + dur * pos_res
                    
                    instrument.notes.append(Note(
                        pitch=pitch,
                        velocity=vel,
                        start=start_tick,
                        end=end_tick
                    ))
                    i += 3
                else:
                    i += 1
            else:
                i += 1
                
        midi_obj.instruments.append(instrument)
        midi_obj.dump(output_path)
