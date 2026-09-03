import uvicorn
from typing import Optional, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid


from experiment.frames import calculate_folder_frame_count

app = FastAPI()

origins = ["*"]

VIDEODIR = "videos/"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/SlowFastResult")
def SlowFastResult(file: UploadFile):
    data = json.load(file.file.read())
    return {"content": data, "filename": file.filename}


@app.get("/experiment/folder_frame_count/{folder_path}")
def cal_folder_frame_count(folder_path: str):
    frames_count = calculate_folder_frame_count(folder_path)
    return {"frames": frames_count}


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # save the file
    with open(f"{VIDEODIR}{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
