
#!/usr/bin/env python3
'''
Created on Nov 08, 2019
@author: Francois Masson

Usage:
  test_net [image1] [image2]

Options:
image1 & image2          path to the images
'''

import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from test_helper import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec
import textwrap as tw

def show(image_1,image_2,similarity, dissimilarity, euc, my_logits):
    fig = plt.figure()
    plt.title(('Similarity: %f, Dissimilarity: %f\nEuclidean Dist: %f, Logits: %f' % (similarity, dissimilarity, euc, my_logits)), loc='center')
    if my_logits > 0.0:
        textstr = 'Similar'
        props = dict(boxstyle='round', facecolor='green', alpha=0.5)
        fig_txt = tw.fill(tw.dedent(textstr), width=80)
        plt.figtext(0.51, 0.05, fig_txt, horizontalalignment='center',
                    fontsize=12, multialignment='center',
                    bbox=dict(boxstyle="round", facecolor='green',
                                ec="0.5", pad=0.5, alpha=1), fontweight='bold')
    else:
        textstr = 'Dissimilar'
        props = dict(boxstyle='round', facecolor='red', alpha=0.5)
        fig_txt = tw.fill(tw.dedent(textstr), width=80)
        plt.figtext(0.51, 0.05, fig_txt, horizontalalignment='center',
                    fontsize=12, multialignment='center',
                    bbox=dict(boxstyle="round", facecolor='red',
                                ec="0.5", pad=0.5, alpha=1), fontweight='bold')



    plt.axis('off')
    ax1 = fig.add_subplot(1, 2, 1)
    l_im = np.array(image_1)[0]
    ax1.imshow(l_im)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    r_im = np.array(image_2)[0]
    ax2.imshow(r_im)
    ax2.axis('off')


    plt.show()

def main():

    parser = argparse.ArgumentParser(description='Test images')
    parser.add_argument('img1', type=str, help='Path to image 1')
    parser.add_argument('img2', type=str, help='Path to image 2')
    arg = parser.parse_args()

    img_one_path = arg.img1
    img_two_path = arg.img2

    img1 = Image.open(img_one_path)
    img2 = Image.open(img_two_path)

    left_input_im = tf.placeholder(tf.float32, [None, 256, 128, 3], 'left_input_im')
    right_input_im = tf.placeholder(tf.float32, [None, 256, 128, 3], 'right_input_im')
    left_label = tf.placeholder(tf.float32, [None, ], 'left_label')
    right_label = tf.placeholder(tf.float32, [None, ], 'right_label')

    ## Check size dimension
    print(np.shape(img1), np.shape(img2))
    print(np.shape(left_input_im), np.shape(right_input_im))

    if np.shape(img1)[0] != np.shape(left_input_im)[1] and np.shape(img1)[1] != np.shape(left_input_im)[2]:
        img1 = img1.resize((np.shape(left_input_im)[2], np.shape(left_input_im)[1]), Image.NEAREST) 
    if np.shape(img2)[0] != np.shape(right_input_im)[1] and np.shape(img2)[1] != np.shape(right_input_im)[2]:
        img2 = img1.resize((np.shape(right_input_im)[2], np.shape(right_input_im)[1]), Image.NEAREST) 
    
    img1 = np.array(img1)[np.newaxis, :, :, :] # Tensor compatible
    img2 = np.array(img2)[np.newaxis, :, :, :]
    
    logits, model_left, model_right = inference(left_input_im, right_input_im)

    global_step = tf.Variable(0, trainable=False)
    global_init = tf.variables_initializer(tf.global_variables())

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(global_init)
        ckpt = tf.train.get_checkpoint_state("model")
        saver.restore(sess, "model_siamese/model.ckpt")

        my_logits, model_lf, model_rg = sess.run([logits, model_left, model_right], \
                                                 feed_dict={left_input_im: img1, right_input_im: img2})

        print(my_logits)
        print(np.shape(model_lf))
        print(np.shape(model_rg))

        lft = np.array(model_lf[0])
        rgt = np.array(model_rg[0])
        l = lft - rgt

        distance = np.sqrt(np.sum((l) ** 2))
        similarity = my_logits * np.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - np.array(my_logits[0])) * np.square(np.max((0.5 - distance),
                                                                        0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = np.mean(dissimilarity + similarity) / 2
        print('distance : ', distance)
        print('similarity : ', similarity)
        print('dissimilarity : ', dissimilarity)
        print('similarity_loss : ', similarity_loss)

        dist = cdist(model_lf, model_rg, 'cosine')
        print('Pairwise distance : ', dist)
        euc = np.linalg.norm(model_lf - model_rg)
        print('euc : ', euc)

        show(img1,img2,similarity, dissimilarity, euc, my_logits)

if __name__ == '__main__':
    main()


