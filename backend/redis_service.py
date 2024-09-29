import os
from datetime import datetime, UTC

import redis

REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT'))


class RedisService:
    def __init__(self):
        self.client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    def update_record(self, uuid: str, objects: dict):
        self.client.hset(name=uuid, mapping={**objects, "created_at": datetime.now(UTC).isoformat()})

    def get_record(self, uuid):
        return self.client.hgetall(uuid)
