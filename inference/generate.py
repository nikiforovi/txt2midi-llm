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
                
                # CRITICAL: Ensure pitch is within valid MIDI range [0, 127]
                note.pitch = max(0, min(127, note.pitch))
        return midi_obj

    @torch.no_grad()
    def generate(
        self, 
        text_prompt: str, 
        tempo: int = 120, 
        mode: str = "major", 
        chromaticity: float = 0.0,
        max_len: int = 512, 
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> List[int]:
        """v2 Generation with advanced sampling (Top-P, Top-K, Repetition Penalty)."""
        
        # Prepare conditioning inputs
        tempo_tensor = torch.tensor([[float(tempo) / 300.0]], device=self.device)
        chrom_tensor = torch.tensor([[chromaticity]], device=self.device)
        
        global_context = self.prompt_encoder([text_prompt], [mode], tempo_tensor, chrom_tensor)
        
        generated = [self.tokenizer.token_to_id["Bar_None"]]
        
        for _ in range(max_len):
            input_tensor = torch.tensor([generated]).to(self.device)
            logits = self.music_model(input_tensor, global_context)
            
            # Get latest logits
            next_token_logits = logits[0, -1, :]
            
            # 1. Apply Frequency-based Repetition Penalty
            if repetition_penalty != 1.0:
                # Track frequencies in the last 64 tokens
                window = generated[-64:]
                from collections import Counter
                counts = Counter(window)
                
                for token, count in counts.items():
                    # Targeted Penalty: Only penalize pitch repetitions
                    token_name = self.tokenizer.id_to_token.get(token, "")
                    if "Note_Pitch" not in token_name:
                        continue # Don't penalize rhythm, duration, velocity, or instruments
                        
                    # Scale penalty by frequency: penalty ^ count
                    effective_penalty = repetition_penalty ** count
                    if next_token_logits[token] > 0:
                        next_token_logits[token] /= effective_penalty
                    else:
                        next_token_logits[token] *= effective_penalty
            
            # 2. Temperature scaling
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            # 3. Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            # 4. Top-P (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 5. Sample
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

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="txt2midi v2: AI Music Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the music")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="generated_output.mid", help="Output MIDI file path")
    parser.add_argument("--tempo", type=int, default=120, help="Tempo (BPM)")
    parser.add_argument("--mode", type=str, default="major", help="Musical mode (major, minor, etc.)")
    parser.add_argument("--root", type=str, default="C", help="Root note (C, D, etc.)")
    parser.add_argument("--len", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--strict", action="store_true", help="Snap notes to the chosen scale")
    
    args = parser.parse_args()
    
    # Load config automatically
    config_path = "configs/model_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"--- txt2midi v2 Inference ---")
    print(f"Loading checkpoint: {args.checkpoint}")
    generator = MusicGeneratorV2(args.checkpoint, config)
    
    print(f"Generating music for prompt: '{args.prompt}'...")
    generator.generate_to_file(
        args.prompt, 
        args.output, 
        tempo=args.tempo, 
        mode=args.mode, 
        root=args.root,
        max_len=args.len,
        temperature=args.temp,
        strict_scale=args.strict
    )
    
    print(f"✅ Success! MIDI saved to: {args.output}")
