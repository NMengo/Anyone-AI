import time
import tensorflow
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input 
from tensorflow.keras.preprocessing import image
import settings
import json
import redis
import os


db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

model = ResNet50(include_top=True, weights='imagenet')

def predict(image_name):
    img_path = os.path.join(settings.UPLOAD_FOLDER, image_name)
    img = image.load_img(img_path, target_size=(224, 224))
    processed_img = image.img_to_array(img)
    processed_img = np.expand_dims(processed_img, axis=0)
    processed_img = preprocess_input(processed_img)

    predictions = model.predict(processed_img)
    results = decode_predictions(predictions, top=1)[0][0]

    return tuple([str(results[1]), round(float(results[2]),4)])


def classify_process():
    while True:
        time.sleep(settings.SERVER_SLEEP)
        
        queue_name, msg = db.brpop(settings.REDIS_QUEUE)
        msg = json.loads(msg)
        image_name = msg['image_name']

        pred_class, pred_score = predict(image_name)

        output_msg = {
            'prediction': pred_class.upper(), 
            'score': round(pred_score, 2)
            }

        db.set(msg['id'], json.dumps(output_msg))


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
