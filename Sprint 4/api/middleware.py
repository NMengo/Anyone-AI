import time
import redis
import settings
import uuid
import json

db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

def model_predict(image_name):
    job_id = str(uuid.uuid4())
    job_data = {
        'id': job_id,
        'image_name': image_name
    }

    db.rpush(settings.REDIS_QUEUE, json.dumps(job_data))

    # Loop until we receive the response from our ML model
    while True:
        time.sleep(settings.API_SLEEP)
        if db.get(job_id):
            output_ = json.loads(db.get(job_id))
            db.delete(job_id)
            break

    return tuple([output_['prediction'], output_['score']])
