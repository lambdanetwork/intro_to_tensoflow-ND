import argparse
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import json

parser = argparse.ArgumentParser(description='')
parser.add_argument('image_path', type=str,
                    help='image path')

parser.add_argument('model_path', type=str,
                    help='model path')

parser.add_argument('--top_k', type=int, help='number of k')
parser.add_argument('--category_names', type=str, help='path to class_names in json')

# destructure parse_args()
args = parser.parse_args()
image_path = args.image_path
model_path = args.model_path
top_k = args.top_k if args.top_k != None else 1

# load model
model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# load json
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

def process_image(img):
    image = np.squeeze(img)
    # This dataset should be resize to 224 x 224
    image = tf.image.resize(image, (224, 224))
    image = image/255
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    probs, indices = tf.math.top_k(prediction, top_k)
    probs = probs.numpy()[0]
    classes = []    
    for i in indices.numpy()[0]:
        class_name = class_names[str(i)]
        classes.append(class_name)

    print('classes: ', classes)
    print('probs: ', probs)
    return probs, classes



predict(image_path, model, top_k)