# https://www.datacamp.com/community/tutorials/tensorflow-tutorial#basics
import os.path
from skimage import data, io, filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage import transform
import random


#################################################################################
def load_data(data_directory):
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


# images = ['1.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm', '2.ppm']
# labels = ['1']


ROOT_PATH = "data/images/"
train_data_directory = os.path.join(ROOT_PATH, "brands")
images, labels = load_data(train_data_directory)
#print(images)
#print(labels)
# exit()
#################################################################################
# Determine the (random) indexes of the images that you want to see
traffic_signs = [1, 2, 3, 4]

# Fill out the subplots with the random images that you defined
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    # plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

# plt.show()

#################################################################################
# # Get the unique labels
# unique_labels = set(labels)
#
# # Initialize the figure
# plt.figure(figsize=(15, 15))
#
# # Set a counter
# i = 1
#
# # For each unique label,
# for label in unique_labels:
#     # You pick the first image for each label
#     image = images[labels.index(label)]
#     # Define 64 subplots
#     plt.subplot(8, 8, i)
#     # Don't include axes
#     plt.axis('off')
#     # Add a title to each subplot
#     plt.title("Label {0} ({1})".format(label, labels.count(label)))
#     # Add 1 to the counter
#     i += 1
#     # And you plot this first image
#     plt.imshow(image)
#
# # Show the plot
# plt.show()

#################################################################################
# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]

#################################################################################
# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

#################################################################################
# traffic_signs = [1, 2, 3, 4]
#
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i + 1)
#     plt.axis('off')
#     plt.imshow(images28[traffic_signs[i]], cmap="gray")
#     plt.subplots_adjust(wspace=0.5)
#
# # Show the plot
# plt.show()

#################################################################################
# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 100, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                     logits=logits))
# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# print(accuracy)

#################################################################################
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

#################################################################################
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print('DONE WITH EPOCH')

#################################################################################
# tf.set_random_seed(1234)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(201):
#         _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
#         if i % 10 == 0:
#             print("Loss: ", loss)
#################################################################################
# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]
# print(sess)
# exit()
# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()
#################################################################################

#################################################################################
# Close the session
sess.close()
