from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException
from datasets import load_dataset
import soundfile as sf
import os
import shutil
from functools import lru_cache
import datetime
from transformers import pipeline as pipeline2
from fastapi.responses import FileResponse
from pydantic import BaseModel


class TextToSpeech(BaseModel):
    text: str

torch_dtype = torch.float32

model_id = "distil-whisper/distil-medium.en" if os.getenv("MODEL_ID") is None else os.getenv("MODEL_ID")


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)


processor = AutoProcessor.from_pretrained(model_id)

pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=0,
    batch_size=1,
    torch_dtype=torch_dtype,
)


synthesiser = pipeline2("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

app = FastAPI()


@lru_cache()
def get_model():
    return pipeline


@lru_cache
def get_synthesiser():
    return synthesiser

@app.post('/v1/audio/transcriptions')
async def transcriptions(file: UploadFile = File(...)):

    if file is None: 
        raise HTTPException(status_code=400, detail="No file provided")
    

    fileObj = file.file
    temp_file_name = f"/tmp{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    temp_file_path = f"/tmp/{temp_file_name}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(fileObj, buffer)

    pipeline = get_model()


    transcription = pipeline(temp_file_path, return_timestamps="word" if os.getenv("RETURN_TIMESTAMPS") == "word" else True)


    os.remove(temp_file_path)
    
    return transcription


@app.post('/v1/audio/speech')
async def speech(audio: TextToSpeech):
    speech = get_synthesiser()(audio.text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    return FileResponse("speech.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6391)