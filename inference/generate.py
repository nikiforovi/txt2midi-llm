import torch
import os
import sys
import torch.nn.functional as F
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.getcwd())

from model.music_transformer import MusicTransformer
from model.prompt_encoder import PromptEncoderV2
from tokenizer.midi_tokenizer import MIDITokenizer
from prompt.prompt_generator import PromptGenerator

class MusicGeneratorV2:
    """v2 Inference: Multi-factor control and musical post-processing."""
    
    def __init__(self, checkpoint_path: str, config: dict, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = config
        
        self.tokenizer = MIDITokenizer(config)
        
        # Load v2 model components
        self.prompt_encoder = PromptEncoderV2(embedding_dim=config['model']['hidden_size']).to(self.device)
        self.music_model = MusicTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config['model']['hidden_size'],
            nhead=config['model']['heads'],
            num_layers=config['model']['layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            max_seq_len=config['model']['context_length']
        ).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.music_model.load_state_dict(checkpoint['music_model'])
            self.prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
        
        self.music_model.eval()
        self.prompt_encoder.eval()

    def snap_to_scale(self, midi_obj, root: str, mode: str):
        """Snaps all generated notes to the nearest note in the given scale."""
        # Scale definitions from midi_analysis
        KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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
        
        degrees = MODE_DEGREES.get(mode, [0, 2, 4, 5, 7, 9, 11])
        root_idx = KEYS.index(root)
        allowed_pitches = [(root_idx + d) % 12 for d in degrees]
        
        for inst in midi_obj.instruments:
            for note in inst.notes:
                pitch_class = note.pitch % 12
                if pitch_class not in allowed_pitches:
                    # Find nearest allowed pitch
                    diffs = [(p - pitch_class + 6) % 12 - 6 for p in allowed_pitches]
                    closest_diff = min(diffs, key=abs)
                    note.pitch += closest_diff
        return midi_obj

    @torch.no_grad()
    def generate(
        self, 
        text_prompt: str, 
        tempo: int = 120, 
        mode: str = "major", 
        chromaticity: float = 0.0,
        max_len: int = 512, 
        temperature: float = 1.0
    ) -> List[int]:
        """v2 Generation with full conditioning."""
        
        # Prepare conditioning inputs
        tempo_tensor = torch.tensor([[float(tempo) / 300.0]], device=self.device)
        chrom_tensor = torch.tensor([[chromaticity]], device=self.device)
        
        global_context = self.prompt_encoder([text_prompt], [mode], tempo_tensor, chrom_tensor)
        
        generated = [self.tokenizer.token_to_id["Bar_None"]]
        
        for _ in range(max_len):
            input_tensor = torch.tensor([generated]).to(self.device)
            logits = self.music_model(input_tensor, global_context)
            
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            if next_token == self.tokenizer.token_to_id["Pad_None"]:
                break
                
        return generated

    def generate_to_file(
        self, 
        text_prompt: str, 
        output_path: str, 
        tempo: int = None, 
        mode: str = "major", 
        root: str = "C",
        strict_scale: bool = False,
        **kwargs
    ):
        """High-level v2 generation with post-processing."""
        # Use provided tempo or fallback to 120 for conditioning
        cond_tempo = tempo if tempo else 120
        
        token_ids = self.generate(text_prompt, tempo=cond_tempo, mode=mode, **kwargs)
        
        # Decode to MIDI object (internal representation)
        import miditoolkit
        # Create a temporary file to decode into
        tmp_path = "tmp_output.mid"
        self.tokenizer.decode(token_ids, tmp_path, target_bpm=tempo)
        
        midi_obj = miditoolkit.midi.parser.MidiFile(tmp_path)
        
        # Scale Snapping
        if strict_scale:
            midi_obj = self.snap_to_scale(midi_obj, root, mode)
            
        midi_obj.dump(output_path)
        if os.path.exists(tmp_path): os.remove(tmp_path)
        
        return output_path
