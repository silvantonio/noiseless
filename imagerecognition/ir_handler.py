import os.path
from skimage import data, io, filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage import transform
import random

class GeneralIRHandler:
    data_root = "data/logos/"

    def __init__(self):
        print('ola')

    def load_data(self, data_directory):
        directories = [d for d in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, d))]
        labels = []
        images = []
        for d in directories:
            label_directory = os.path.join(data_directory, d)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith(".ppm")]
            # print(file_names)
            # print(d)
            for f in file_names:
                images.append(data.imread(f))
                labels.append(int(d))
        return images, labels

    def predict(self, image_url):
        #print(image_url)
        train_data_directory = self.data_root
        images, labels = self.load_data(train_data_directory)
        # Rescale the images in the `images` array
        images28 = [transform.resize(image, (28, 28)) for image in images]
        # Convert `images28` to an array
        images28 = np.array(images28)
        # Convert `images28` to grayscale
        images28 = rgb2gray(images28)
        # Initialize placeholders
        x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
        y = tf.placeholder(dtype=tf.int32, shape=[None])
        # Flatten the input data
        images_flat = tf.contrib.layers.flatten(x)
        # Fully connected layer
        logits = tf.contrib.layers.fully_connected(images_flat, 100, tf.nn.relu)
        # Define a loss function
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        # Define an optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        # Convert logits to label indexes
        correct_pred = tf.argmax(logits, 1)
        # Define an accuracy metric
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Start session
        tf.set_random_seed(1234)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        imagedata= io.imread(image_url)
        # Rescale the images in the `images` array
        images28 = transform.resize(imagedata, (28, 28))
        # Convert `images28` to an array
        images28 = np.array(images28)
        # Convert `images28` to grayscale
        images28 = rgb2gray(images28)
        # predict
        predicted = sess.run([correct_pred], feed_dict={x: [images28]})[0]
        print(predicted)
