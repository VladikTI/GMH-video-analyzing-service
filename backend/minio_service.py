import io
import os

from minio import Minio

from models import FileRecord

MINIO_HOST = os.getenv('MINIO_HOST')
MINIO_PORT = os.getenv('MINIO_PORT')
MINIO_USER = os.getenv('MINIO_ROOT_USER')
MINIO_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')


class MinioService:

    def __init__(self, bucket_name):
        self.client = Minio(
            endpoint=MINIO_HOST + ':' + MINIO_PORT,
            access_key=MINIO_USER,
            secret_key=MINIO_PASSWORD,
            secure=False
        )

        self.bucket_name = self.create_bucket(bucket_name)

    def create_bucket(self, bucket_name):
        found = self.client.bucket_exists(bucket_name)
        if not found:
            self.client.make_bucket(bucket_name)
        return bucket_name

    def put_object(self, file: FileRecord, content):
        self.client.put_object(
            self.bucket_name,
            str(file.id),
            data=io.BytesIO(content),
            length=-1,
            part_size=10 * 1024 * 1024,
            content_type=file.content_type
        )

    def get_object(self, object_name):
        response = None
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.read()
        finally:
            response.close()
            response.release_conn()
