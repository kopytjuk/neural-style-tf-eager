import tensorflow as tf
from utils import gram_matrix


def get_content_loss(content_features, target_features):
	"""Calculates an elementwise euclidean distance between calculated VGG19 features from the content image and target image.
	Lower cost increases the similarity between generated and content image.
	
	Args:
		content_features (tf.Tensor): Tensor (layer output) given the content image
		target_features (tf.Tensor): Tensor (layer output) given the generated image
	"""
	return tf.reduce_mean(tf.square(content_features - target_features))


def get_style_loss(base_style, gram_target):
	"""Calculates style loss of the generated image given a precomputed gram target from style image.
	
	Args:
		base_style (tf.Tensor): (style-)layer output given generated image
		gram_target (tf.Tensor): Calculated Gram matrix from style-image (style layer output)
	
	Returns:
		tf.Tensor: Style loss
	"""

	# height, width, num filters of each layer
	# We scale the loss at a given layer by the size of the feature map and the number of filters
	height, width, channels = base_style.get_shape().as_list()
	gram_style = gram_matrix(base_style)

	return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)
