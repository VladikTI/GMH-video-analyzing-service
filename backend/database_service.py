import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import joinedload, selectinload

from redis_service import RedisService
from models import Base, FileRecord

DATABASE_HOST = os.getenv("POSTGRES_HOST")
DATABASE_PORT = os.getenv("POSTGRES_PORT")
DATABASE_USER = os.getenv("POSTGRES_USER")
DATABASE_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DATABASE_NAME = os.getenv("POSTGRES_DB")
DATABASE_URL = f'postgresql+asyncpg://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'


class DatabaseService:
    def __init__(self):
        self.engine = create_async_engine(
            DATABASE_URL,
            echo=True,
        )

        self.async_session = async_sessionmaker(bind=self.engine, expire_on_commit=False, class_=AsyncSession)

    async def get_session(self) -> AsyncSession:
        async with self.async_session() as session:
            yield session

    async def create_tables(self):
        async with self.async_session():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)


async def add_record(session: AsyncSession, record):
    session.add(record)
    await session.commit()
    await session.refresh(record)


async def update_record(session: AsyncSession, record):
    await session.commit()
    await session.refresh(record)


async def get_videos_with_filters(session: AsyncSession, rediss: RedisService, filters=None):
    if filters is None:
        filters = []

    query = select(FileRecord).options(
        selectinload(FileRecord.key_intervals),
        selectinload(FileRecord.tag),
        selectinload(FileRecord.tonalities)
    )

    for filter_item in filters:
        query = query.where(FileRecord.tag.category == filter_item)

    result = await session.execute(query)
    files = result.scalars().all()

    videos = []
    for file in files:

        _task = rediss.get_record(file.task_uuid)
        _task = {k.decode('utf-8'): v.decode('utf-8') for k, v in _task.items()}
        status = _task.get('status')

        if status == 'completed':
            video_data = {
                'file_id': file.id,
                'title': file.name,
                'status': status,
                'content_type': file.content_type,
                'details': {
                    'key_intervals': [
                        {
                            "start_time": interval.start_time,
                            "end_time": interval.end_time,
                            "title": interval.title,
                            "interest_point": interval.interest_point,
                            "objects": interval.objects,
                            "events": interval.events,
                            "sounds": interval.sounds,
                            "music": interval.music,
                            "symbols": interval.symbols
                        }
                        for interval in file.key_intervals
                    ],
                    'tags': {
                        'category': file.tag.category,
                        'events': file.tag.events,
                        'sounds': file.tag.sounds,
                        'music': file.tag.music,
                        'symbols': file.tag.symbols,
                        'voice_transcription': file.tag.voice_transcription,
                    },
                    'tonalities': [
                        {
                            'tag': tonality.tag,
                            'type': tonality.type,
                            'tonalities': tonality.tonalities,
                        }
                        for tonality in file.tonalities],
                }
            }
        else:
            video_data = {
                'file_id': file.id,
                'title': file.name,
                'status': status,
                'content_type': file.content_type
            }

        videos.append(video_data)

    return videos


async def get_file_details(session: AsyncSession, file_id):
    result = await session.execute(
        select(FileRecord)
        .options(
            joinedload(FileRecord.key_intervals),
            joinedload(FileRecord.tag),
            joinedload(FileRecord.tonalities)
        )
        .filter(FileRecord.id == file_id)
    )
    file_record = result.scalars().unique().one_or_none()
    return file_record
