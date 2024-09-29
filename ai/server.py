import logging
import os
import sys

import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

from minio_service import MinioService
from ml import load_video
app = FastAPI()

# Configure Gemini API keys
# Folder to temporarily store uploaded files
UPLOAD_DIR = "./uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Gemini model details
generation_config = genai.GenerationConfig(
    response_mime_type="application/json"
)
_minio_service = MinioService('main')


class FileDto(BaseModel):
    file_id: str
    name: str


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


logger = get_logger('openai', logging.DEBUG)


@app.post("/markup-video")
async def analyze_video(file_dto: FileDto):
    # Save the uploaded file temporarily
    file_content = _minio_service.get_object(file_dto.file_id)

    file_location = f"{UPLOAD_DIR}/{file_dto.file_id}_{file_dto.name}"

    with open(file_location, "wb") as buffer:
        buffer.write(file_content)

    logger.debug(f"Saved file to {file_location}")

    load_video(file_content)

    # Return the final response
    return json_response
