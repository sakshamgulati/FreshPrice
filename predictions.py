import requests
import json
from PIL import Image, ImageOps

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def get_img(image_path):
    # Load the model
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
    image = load_img(image_path)
    print("Image loaded")
    # image sizing
    size = (200, 200)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255
    normalized_image_array = np.expand_dims(normalized_image_array, 0)
    print("Image resized")

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    return normalized_image_array


path = "avocado.jpg"
img = get_img(path)

data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
headers = {"content-type": "application/json"}

json_response = requests.post(
    "http://freshprice.herokuapp.com/v1/models/model:predict",
    data=data,
    headers=headers,
)
predictions = json.loads(json_response.text)["predictions"]

print(predictions[0][0])
