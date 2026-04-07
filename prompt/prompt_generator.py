import random
from typing import Dict, Any

class PromptGenerator:
    """Generates synthetic text prompts from MIDI attributes."""
    
    def __init__(self):
        self.templates = [
            "{tempo_label} {density} density {genre} piece in {key} {scale} at {tempo} bpm",
            "{genre} {instrument} in {key} {scale}, {tempo} bpm, {density} density",
            "a {tempo_label} {genre} song in {key} {scale} with {density} note density",
            "{instrument} {genre} music, {key} {scale}, {tempo} bpm"
        ]

    def _normalize_attributes(self, attr: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures attributes are in a consistent format for prompting."""
        norm = attr.copy()
        if "key" in norm and norm["key"]:
            # C# -> C#, Gb -> Gb, Normalize case
            key = norm["key"].strip().capitalize()
            # Standardize notation if needed (e.g. s to #)
            key = key.replace("s", "#")
            norm["key"] = key
            
        if "scale" in norm and norm["scale"]:
            norm["scale"] = norm["scale"].lower()
            
        return norm

    def generate(self, attributes: Dict[str, Any], instrument: str = "piano") -> str:
        """Creates a text prompt based on attributes.
        
        Args:
            attributes: Dict containing tempo, key, scale, density, etc.
            instrument: The primary instrument name.
            
        Returns:
            A string prompt.
        """
        attr = self._normalize_attributes(attributes)
        template = random.choice(self.templates)
        
        # Ensure all required keys exist in attributes
        data = {
            "tempo_label": attr.get("tempo_label", "moderate"),
            "density": attr.get("density", "medium"),
            "genre": attr.get("genre", "pop"),
            "key": attr.get("key", "C"),
            "scale": attr.get("scale", "major"),
            "tempo": attr.get("tempo", 120),
            "instrument": instrument
        }
        
        return template.format(**data)

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Simple rule-based parser for user prompts (Inference).
        In the baseline, this is a placeholder. 
        For a production system, this could use an LLM or a small NER model.
        """
        # Baseline: look for keywords
        attr = {
            "tempo": 120,
            "key": "C",
            "scale": "major",
            "genre": "pop"
        }
        
        # Simple keyword matching
        words = prompt.lower().split()
        
        # Key detection (A-G)
        keys = ["c", "d", "e", "f", "g", "a", "b"]
        for word in words:
            if word in keys:
                attr["key"] = word.upper()
                
        if "minor" in words:
            attr["scale"] = "minor"
        elif "major" in words:
            attr["scale"] = "major"
            
        # BPM detection
        for i, word in enumerate(words):
            if word == "bpm" and i > 0:
                try:
                    attr["tempo"] = int(words[i-1])
                except:
                    pass
            elif word.isdigit():
                val = int(word)
                if 40 < val < 250:
                    attr["tempo"] = val

        return attr
