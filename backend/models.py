from sqlalchemy import Column, Integer, String, ForeignKey, Text, ARRAY
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class FileRecord(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text)
    content_type = Column(String)
    task_uuid = Column(String)


class KeyIntervalRecord(Base):
    __tablename__ = 'key_intervals'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    start_time = Column(Text, nullable=False)
    end_time = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    interest_point = Column(Text)
    objects = Column(ARRAY(Text))
    events = Column(ARRAY(Text))
    sounds = Column(ARRAY(Text))
    music = Column(ARRAY(Text))
    symbols = Column(ARRAY(Text))

    file_record = relationship('FileRecord', back_populates='key_intervals')


class TagRecord(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    category = Column(Text, nullable=False)
    events = Column(ARRAY(Text))
    sounds = Column(ARRAY(Text))
    music = Column(ARRAY(Text))
    symbols = Column(ARRAY(Text))
    voice_transcription = Column(Text)

    file_record = relationship('FileRecord', back_populates='tag', uselist=False)


class TonalityRecord(Base):
    __tablename__ = 'tonalities'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    tag = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    tonalities = Column(ARRAY(Text))

    file_record = relationship('FileRecord', back_populates='tonalities')


FileRecord.key_intervals = relationship(
    'KeyIntervalRecord',
    back_populates='file_record',
    cascade="all, delete-orphan"
)
FileRecord.tag = relationship('TagRecord', back_populates='file_record', uselist=False)
FileRecord.tonalities = relationship(
    'TonalityRecord',
    back_populates='file_record',
    cascade="all, delete-orphan"
)
