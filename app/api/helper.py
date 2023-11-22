import tensorflow as tf

from tensorflow.keras.preprocessing import image

list_of_result = ["paper", "rock", "scissors"]

def helper(image_path, shape=(150, 150)):
    load_image = tf.io.decode_image(image_path, channels=3)
    image_array = image.img_to_array(load_image)
    img = tf.image.resize(image_array, shape)
    result = tf.expand_dims(img, axis=0)
    return result

def load_model(rimages):
    modelapi = tf.keras.models.load_model("modelapi/")
    images = helper(rimages)
    logits = modelapi.predict(images, verbose=0)
    result = list_of_result[tf.argmax(logits[0], 0)]
    return result