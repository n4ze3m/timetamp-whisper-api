from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException

import os
import shutil
from functools import lru_cache
import datetime


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
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
)


app = FastAPI()


@lru_cache()
def get_model():
    return pipeline




@app.post('/v1/audio/transcriptions')
async def transcriptions(file: UploadFile = File(...)):

    if file is None: 
        raise HTTPException(status_code=400, detail="No file provided")
    

    file_name = file.filename
    fileObj = file.file

    # Save the file temporarily
    temp_file_name = f"/tmp{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    temp_file_path = f"/tmp/{temp_file_name}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(fileObj, buffer)

    pipeline = get_model()


    transcription = pipeline(temp_file_path, return_timestamps="word" if os.getenv("RETURN_TIMESTAMPS") == "word" else True)


    os.remove(temp_file_path)
    
    return transcription


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6391)