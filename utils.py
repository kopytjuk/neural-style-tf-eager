from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.python.keras.preprocessing import image as kp_image
from PIL import Image
import numpy as np
import tensorflow as tf

def gram_matrix(input_tensor):
	"""Caculates the Gram-Matrix from a layer output (tensor).

	Args:
		input_tensor (tf.Tensor): Layer output
	
	Returns:
		tf.Tensor: Gram matrix
	"""

	# We make the image channels first 
	channels = int(input_tensor.shape[-1])

	# [Wf, Hf, Cn] -> [Wf*Hf, Cn]
	a = tf.reshape(input_tensor, [-1, channels])
	n = tf.shape(a)[0] # Wf*Hf

	gram = tf.matmul(a, a, transpose_a=True)
	return gram / tf.cast(n, tf.float32)

def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img

def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = preprocess_input_vgg19(img)
    return img


def save_image(img_arr, path):
    img = Image.fromarray(img_arr)
    img.save(path)


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
        "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
    
    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x
