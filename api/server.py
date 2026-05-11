from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import yaml
import uuid
import sys
from typing import Optional

# Add project root to sys.path
sys.path.append(os.getcwd())

from inference.generate import MusicGeneratorV2

app = FastAPI(title="txt2midi v2 API")

# Load config
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize generator v2 with dynamic checkpoint selection
def get_latest_checkpoint(directory="checkpoints"):
    import glob
    checkpoints = glob.glob(os.path.join(directory, "*.pth"))
    if not checkpoints:
        return None
    # Sort by modification time to get the newest
    return max(checkpoints, key=os.path.getmtime)

LATEST_CHECKPOINT = get_latest_checkpoint()

if LATEST_CHECKPOINT:
    print(f"--- Loading v2 Model from: {LATEST_CHECKPOINT} ---")
    generator = MusicGeneratorV2(LATEST_CHECKPOINT, config)
else:
    print("WARNING: No checkpoints found in 'checkpoints/' directory. Server will fail on generation.")
    generator = None

class GenerateRequest(BaseModel):
    prompt: str
    tempo: Optional[int] = 120
    mode: Optional[str] = "major"
    root: Optional[str] = "C"
    strict_scale: Optional[bool] = False
    chromaticity: Optional[float] = 0.0
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_len: int = 512

@app.post("/generate")
async def generate_midi(request: GenerateRequest):
    """Generates a MIDI file with v2 parameters."""
    if not generator:
        raise HTTPException(status_code=503, detail="Model generator not initialized. Check if checkpoints exist.")
        
    try:
        output_dir = "data/generated"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4()}.mid"
        file_path = os.path.join(output_dir, filename)
        
        generator.generate_to_file(
            text_prompt=request.prompt,
            output_path=file_path,
            tempo=request.tempo,
            mode=request.mode,
            root=request.root,
            strict_scale=request.strict_scale,
            chromaticity=request.chromaticity,
            max_len=request.max_len,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        return FileResponse(
            path=file_path, 
            filename="generated_music.mid",
            media_type="audio/midi"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": "txt2midi-v2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
