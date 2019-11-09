#!/usr/bin/env python3
'''
Created on Nov 07, 2019
@author: Francois Masson

Usage:
  train_net [folder]

Options:
folder          Your dataset directory (e.g: MARS PATH ROOT)
'''

import tensorflow as tf
import os
import glob
import numpy as np
import argparse
from pathlib import Path

from train_helper import *


def main():
    parser = argparse.ArgumentParser(description='Training options')
    parser.add_argument('path', type=str, help='MARS_DATASET_ROOT path')
    parser.add_argument('--transfer_learning',default=False,type=bool, help='Switch to transfer learning for the training')
    parser.add_argument('--contrastive',default=False,type=bool, help='Switch to transfer learning for the training')
    parser.add_argument('--model_save',default=None,type=str, help='Folder to save the config. By default : ./model_siamese/')
    arg = parser.parse_args()
    print(arg.transfer_learning)
    data_filename = os.path.join(arg.path, 'data_summary.txt')

    with tf.gfile.Open(data_filename, 'r') as f:
        num_validatiaon = f.readline()
        num_dataset = f.readline()

    print('Found %d images in the training data' % (int(num_dataset) - int(num_validatiaon)))
    print('Found %d images in the validataion data' % (int(num_validatiaon)))
    training_data_num = int(num_dataset) - int(num_validatiaon)

    BATCH_SIZE = 32
    num_epochs = 200

    train_record = str(glob.glob(str(Path(arg.path)) + '/*train*.tfrecord')[0])
    val_record = str(glob.glob(str(Path(arg.path)) + '/*validation*.tfrecord')[0])

    train_dataset = combine_dataset(tfrecords_path=train_record,batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=True, transfer=arg.transfer_learning)
    val_dataset = combine_dataset(tfrecords_path=val_record,batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=False, transfer=arg.transfer_learning)
    handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
               handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()

    left, right = iterator.get_next()
    left_input_im, left_label, left_addr = left
    right_input_im, right_label, right_addr = right

    if arg.transfer_learning:
        logits, model_left, model_right = transfer_learning(left_input_im, right_input_im)
    else :
        logits, model_left, model_right = inference(left_input_im, right_input_im)

    if arg.contrastive:    
        contrastive_loss(model_left, model_right, logits, left_label, right_label, margin=0.2, use_loss=True)
    else :
        loss(logits, left_label, right_label)

    total_loss = tf.losses.get_total_loss()
    global_step = tf.Variable(0, trainable=False)

    if arg.transfer_learning:
        updates = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss,global_step = global_step)
    else :
        params = tf.trainable_variables()
        gradients = tf.gradients(total_loss, params)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        updates = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)
        global_init = tf.variables_initializer(tf.global_variables())

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(global_init)

        # setup tensorboard
        if not os.path.exists('train.log'):
            os.makedirs('train.log')
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)

        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(val_iterator.string_handle())

        num_iterations = training_data_num // BATCH_SIZE

        for epoch in range(num_epochs):
            print('epoch : ', epoch, ' / ', num_epochs)
            for iteration in range(num_iterations):
                feed_dict_train = {handle:training_handle}

                loss_train, _, summary_str = sess.run([total_loss, updates, merged], feed_dict_train)
                writer.add_summary(summary_str, epoch)
                print("iteration : %d / %d - Loss : %f" % (iteration, num_iterations, loss_train))

            feed_dict_val = {handle: validation_handle}
            val_loss = sess.run([total_loss], feed_dict_val)
            print('========================================')
            print("epoch : %d - Validation Loss : %f" % (epoch, val_loss[0]))
            print('========================================')

            if arg.model_save is None:
                if not os.path.exists("./model_siamese/"):
                os.makedirs("./model_siamese/")
                saver.save(sess, "model_siamese/model.ckpt")
            else :
                if not os.path.exists(arg.model_save):
                os.makedirs(arg.model_save)
                saver.save(sess, os.path.join(arg.model_save, "/model.ckpt"))

if __name__ == "__main__":
    main()
    