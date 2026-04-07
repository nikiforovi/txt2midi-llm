from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import yaml
import uuid
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from inference.generate import MusicGenerator

app = FastAPI(title="txt2midi API")

# Load config
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize generator
# Note: In a real scenario, you'd point this to your best checkpoint
CHECKPOINT_PATH = "checkpoints/baseline_epoch_50.pth"
generator = MusicGenerator(CHECKPOINT_PATH, config)

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    max_len: int = 512

@app.post("/generate")
async def generate_midi(request: GenerateRequest):
    """Generates a MIDI file from a text prompt."""
    try:
        output_dir = "data/generated"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4()}.mid"
        file_path = os.path.join(output_dir, filename)
        
        generator.generate_to_file(
            text_prompt=request.prompt,
            output_path=file_path,
            max_len=request.max_len,
            temperature=request.temperature
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
    return {"status": "ok", "model": "txt2midi-baseline"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
