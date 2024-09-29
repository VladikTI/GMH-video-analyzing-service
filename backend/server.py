import json
import logging
import os
import sys
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from minio import S3Error
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketDisconnect

from database_service import DatabaseService, add_record, get_file_details, get_videos_with_filters
from minio_service import MinioService
from models import *
from redis_service import RedisService
from connection_manager import ConnectionManager

AI_HOST = os.getenv("AI_HOST")
AI_PORT = os.getenv("AI_PORT")

_minio_service = MinioService('main')
_db_service = DatabaseService()
_redis_service = RedisService()
ws_manager = ConnectionManager()

app = FastAPI(root_path="/api")

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://194.87.26.211"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


logger = get_logger('backend', logging.DEBUG)


class FileDetailsDto(BaseModel):
    id: str


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                 session: AsyncSession = Depends(_db_service.get_session)):
    try:
        await _db_service.create_tables()
        task_id = str(uuid.uuid4())
        new_file = FileRecord(
            content_type=file.content_type,
            name=file.filename,
            task_uuid=task_id,
        )
        await add_record(session, new_file)

        _redis_service.update_record(task_id, {'file_id': new_file.id, 'status': 'uploaded'})

        content = await file.read()
        _minio_service.put_object(new_file, content)

        background_tasks.add_task(process_file, task_id, session, new_file)

        videos = await get_videos_with_filters(session, _redis_service)
        await ws_manager.broadcast(videos)

        return JSONResponse(status_code=201, content={'task_id': task_id})
    except S3Error as e:
        return Response(content=str(e), status_code=400)
    except Exception as e:
        return Response(content=str(e), status_code=500)


@app.get('/details')
async def get_video_details(dto: FileDetailsDto, session: AsyncSession = Depends(_db_service.get_session)):
    try:
        return JSONResponse(content=await get_file_details(session, dto.id), status_code=200)
    except Exception as e:
        return Response(content=str(e), status_code=500)


@app.websocket("/videos/list")
async def videos_list(websocket: WebSocket, session: AsyncSession = Depends(_db_service.get_session)):
    await ws_manager.connect(websocket)
    await _db_service.create_tables()
    videos = await get_videos_with_filters(session, _redis_service)
    await ws_manager.broadcast(videos)
    try:
        while True:
            data = await websocket.receive_json()
            filters = data.get('data', [])
            videos = await get_videos_with_filters(session, _redis_service, filters)
            await ws_manager.broadcast(videos)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/tasks/{task_id}")
async def get_task_status(task_id):
    task = _redis_service.get_record(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task = {k.decode('utf-8'): v.decode('utf-8') for k, v in task.items()}
    return task


async def process_file(task_id, session, file):
    _redis_service.update_record(task_id, {'file_id': file.id, 'status': 'in_progress'})
    videos = await get_videos_with_filters(session, _redis_service)
    await ws_manager.broadcast(videos)

    async with aiohttp.ClientSession() as http_session:
        data = {
            'file_id': str(file.id),
            'name': file.name,
        }

        async with http_session.post(f'http://{AI_HOST}:{AI_PORT}/markup-video', json=data) as response:
            if response.status == 200:
                ai_data = await response.json()
                ai_data = json.loads(ai_data)
            else:
                _redis_service.update_record(task_id, {'file_id': file.id, 'status': 'failed'})
                videos = await get_videos_with_filters(session, _redis_service)
                await ws_manager.broadcast(videos)
                return

    await create_records(session, ai_data, file.id)
    _redis_service.update_record(task_id, {'file_id': file.id, 'status': 'completed'})
    _videos = await get_videos_with_filters(session, _redis_service)
    await ws_manager.broadcast(_videos)


async def create_records(session: AsyncSession, ai_data, file_id):
    logger.debug(ai_data)
    for interval in ai_data.get('key_intervals', []):
        new_interval = KeyIntervalRecord(
            file_id=file_id,
            start_time=interval.get('start_time'),
            end_time=interval.get('end_time'),
            title=interval.get('title'),
            interest_point=interval.get('interest_point'),
            objects=interval.get('objects', []),
            events=interval.get('events', []),
            sounds=interval.get('sounds', []),
            music=interval.get('music', []),
            symbols=interval.get('symbols', []),
        )
        await add_record(session, new_interval)

    _tag = ai_data.get('tags')
    new_tag = TagRecord(
        file_id=file_id,
        category=_tag.get('category'),
        events=_tag.get('events', []),
        sounds=_tag.get('sounds', []),
        music=_tag.get('music', []),
        symbols=_tag.get('symbols', []),
        voice_transcription=_tag.get('voice_transcription', ''),

    )
    await add_record(session, new_tag)

    _tonalities = ai_data.get('tonality_objects')
    for key in _tonalities:
        new_tonality = TonalityRecord(
            file_id=file_id,
            tag=key,
            type='object',
            tonalities=_tonalities[key],
        )
        await add_record(session, new_tonality)

    _tonalities = ai_data.get('tonality_events')
    for key in _tonalities:
        new_tonality = TonalityRecord(
            file_id=file_id,
            tag=key,
            type='event',
            tonalities=_tonalities[key],
        )
        await add_record(session, new_tonality)

    _tonalities = ai_data.get('tonality_sounds')
    for key in _tonalities:
        new_tonality = TonalityRecord(
            file_id=file_id,
            tag=key,
            type='sound',
            tonalities=_tonalities[key],
        )
        await add_record(session, new_tonality)

    _tonalities = ai_data.get('tonality_music')
    for key in _tonalities:
        new_tonality = TonalityRecord(
            file_id=file_id,
            tag=key,
            type='music',
            tonalities=_tonalities[key],
        )
        await add_record(session, new_tonality)

    _tonalities = ai_data.get('tonality_symbols')
    for key in _tonalities:
        new_tonality = TonalityRecord(
            file_id=file_id,
            tag=key,
            type='symbol',
            tonalities=_tonalities[key],
        )
        await add_record(session, new_tonality)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
