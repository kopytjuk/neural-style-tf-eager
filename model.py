import tensorflow as tf
from tensorflow.python.keras import models

from utils import load_and_process_img

from losses import get_content_loss, get_style_loss, gram_matrix

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
    """ Creates our model with access to intermediate layers. 

    This function will load the VGG19 model and access the intermediate layers. 
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the VGG model. 

    Returns:
    returns a keras model that takes image inputs and outputs the style and 
        content intermediate layers. 
    """
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers (output tensors)
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model 
    return models.Model(vgg.input, model_outputs)


def get_features_forward_pass(model, content_path, style_path):

    image_content_tensor = load_and_process_img(content_path)
    image_style_tensor = load_and_process_img(style_path)


    model_input = tf.concat([image_style_tensor, image_content_tensor])

    # get outputs from both style and content images
    outputs_model = model(model_input)

    # o[0] to get the first image from batch (style image)
    style_features = get_style_features_from_model_output(outputs_model, sample_slice=0)

    # o[0] to get the second image from batch (content image)
    content_features = get_content_features_from_model_output(outputs_model, sample_slice=1)

    return style_layer_outputs, content_layer_outputs


def get_style_features_from_model_output(out, sample_slice=0):
    return [o[sample_nr] for o in out[:num_style_layers]]

def get_content_features_from_model_output(out, sample_slice=0):
    return [o[1] for o in out[num_style_layers:]]


def compute_overall_loss(model, loss_weights, init_image, gram_style_image, content_image_features):
    """Compute overall loss from single training iteration. We feed the generates image X throuth the
    model's forward pass and calculate the loss given the style features from style image and content features given
    the content image.
    
    Args:
        model (object): Instance with an implemented forward pass to extract generated style and content features
        loss_weights (tuple): The weights of each contribution of each loss component. len(loss_weights) == 2
        init_image (tf.Variable): Tensor of generated image
        gram_style_features (tf.Tensor): Tensor of calculated Gram features
        content_features (list): Tensor of precalculated content features (layers)
    """
    
    # unpack loss weights
    style_weight, content_weight = loss_weights

    # get the features of generated image X, F = VGG(X)
    gen_features = model(init_image)

    gen_style_layers = get_style_features_from_model_output(gen_features, sample_slice=0)
    gen_content_layers = get_content_features_from_model_output(gen_features, sample_slice=0)

    style_score = 0 # works only in eager mode
    weight_per_style_layer = 1.0 / float(num_style_layers)
    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    for s in range(num_style_layers):
        gen_gram_s = gram_matrix(gen_style_layers[s])
        style_loss_s = get_style_loss(gen_gram_s, gram_style_image[s])
        style_score += weight_per_style_layer*style_loss_s


    content_score = 0

    weight_per_content_layer = 1.0 / float(num_content_layers)
    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    for c in range(num_content_layers):
        content_loss_c = weight_per_content_layer*get_content_loss(gen_content_layers[c], content_image_features[c])
        content_score += content_loss_c

    style_score_weighted = style_weight*style_score
    content_score_weighted = content_weight*content_score
    
    L =  style_score_weighted + content_score_weighted

    return L, style_score_weighted, content_score_weighted
