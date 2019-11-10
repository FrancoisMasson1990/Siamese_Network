import tensorflow as tf
import os
import glob
import numpy as np

## Library used for the transfer learning
import tensornets as nets

def model(input, reuse=False):
    print(np.shape(input))
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 256, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        net = tf.contrib.layers.flatten(net)
        print(np.shape(net))

        net = tf.layers.dense(net, 4096, activation=tf.sigmoid)
        print(np.shape(net))

    return net

def inference(left_input_image, right_input_image):
    """
	left_input_image: 3D tensor input
	right_input_image: 3D tensor input
	label: 1 if images are from same category. 0 if not.
	"""
    with tf.variable_scope('feature_generator', reuse=tf.AUTO_REUSE) as sc:

        left_features = model(tf.layers.batch_normalization(tf.divide(left_input_image, 255.0)))
        right_features = model(tf.layers.batch_normalization(tf.divide(right_input_image, 255.0)))

    merged_features = tf.abs(tf.subtract(left_features, right_features))
    logits = tf.contrib.layers.fully_connected(merged_features, num_outputs=1, activation_fn=None)
    logits = tf.reshape(logits, [-1])
    return logits, left_features, right_features

def transfer_learning(left_input_image, right_input_image):
    """
	left_input_image: 3D tensor input
	right_input_image: 3D tensor input
	label: 1 if images are from same category. 0 if not.
	"""

    left_features = nets.VGG19(left_input_image,is_training=True,reuse=tf.AUTO_REUSE)
    left_features.pretrained()
    right_features = nets.VGG19(right_input_image, is_training=True,reuse=tf.AUTO_REUSE)
    right_features.pretrained()

    merged_features = tf.abs(tf.subtract(left_features, right_features))
    logits = tf.contrib.layers.fully_connected(merged_features, num_outputs=1, activation_fn=None)
    logits = tf.reshape(logits, [-1])
    return logits, left_features, right_features
