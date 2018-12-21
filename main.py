import numpy as np
import time
import functools
import datetime
import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from model import get_model, get_features_forward_pass, get_style_features_from_model_output, get_content_features_from_model_output, get_overall_loss
from utils import gram_matrix, load_and_process_img, deprocess_img, save_image
from misc import create_training_gif

print('Using tensorflow', tf.__version__)

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


if __name__ == '__main__':

    content_image_path = 'img-raw/content2.png'
    style_image_path = 'img-raw/style4.jpg'
    training_base_path = "results"
    loss_weights = (5e-3, 5e3)
    n_iterations = 1000
    learning_rate = 8
    epochs_early_stopping = 10
    save_each_epoch = 10

    # create a training directory
    dt_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_path = os.path.join(training_base_path, dt_str)
    if not os.path.exists(training_path):
        os.makedirs(training_path)

    # create some folders for training tracking
    os.makedirs(os.path.join(training_path, 'img'))
    os.makedirs(os.path.join(training_path, 'metrics'))

    model = get_model()

    # Set initial image
    init_image = load_and_process_img(content_image_path)
    X = tfe.Variable(init_image, trainable=True, dtype=tf.float32)

    # run the model once to get the target style and content features
    style_layer_outputs, content_layer_outputs = get_features_forward_pass(model, content_image_path, style_image_path)

    style_image_matrices = [gram_matrix(s) for s in style_layer_outputs]

    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None

    # for early stopping
    iters_wo_improvement = 0

    loss_history = {"L": list(), "L_style": list(), "L_content": list()}

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -1*norm_means
    max_vals = 255 - norm_means 

    for i in range(n_iterations):

        with tf.GradientTape() as tape:
            L, L_style, L_content = get_overall_loss(model, loss_weights, X, style_image_matrices, content_layer_outputs)

        # log history and losses / todo, maybe dvc?
        loss_history['L'].append(L.numpy())
        loss_history['L_style'].append(L_style.numpy())
        loss_history['L_content'].append(L_content.numpy())

        # implement early stopping
        if L.numpy() >= best_loss:
            iters_wo_improvement += 1
        else:
            best_loss = L.numpy()
            best_img = deprocess_img(X.numpy())
        
        if iters_wo_improvement > epochs_early_stopping:
            print('Stopped training after %d iters.'%(i))
            break

        if i%save_each_epoch == 0:
            save_image(deprocess_img(X.numpy()), os.path.join(training_path, "img/iter_%05d.png"%i))

        print('iter %05d: L=%.2E, L_style=%.2E, L_content=%.2E'%(i, L.numpy(), L_style.numpy(), L_content.numpy()))
        
        dL_dX = tape.gradient(L, X)
        opt.apply_gradients([(dL_dX, X)], global_step=tf.train.get_or_create_global_step())

        # in order to prevent large image values
        clipped = tf.clip_by_value(X, min_vals, max_vals)
        X.assign(clipped)

    create_training_gif(os.path.join(training_path, "training.gif"), content_image_path, style_image_path, os.path.join(training_path, "img"))
