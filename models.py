from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random


def linear_unit(x, W, b):
  return tf.matmul(x, W) + b

class ModelPart0:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        output = 2
        self.W1 = tf.Variable(tf.random.truncated_normal([input, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")


        self.trainable_variables = [self.W1, self.B1]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

    # this reshape "flattens" the image data
        inputs = np.reshape(inputs, [inputs.shape[0],-1])
        x = linear_unit(inputs, self.W1, self.B1)
        return x


class ModelPart1:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input_layer_1 = 32 * 32 * 3
        output_layer_1 = 256
        input_layer_2 = 256
        output_layer_2 = 2

        self.W1 = tf.Variable(tf.random.truncated_normal([input_layer_1, output_layer_1],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output_layer_1],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")
        self.W2 = tf.Variable(tf.random.truncated_normal([input_layer_2, output_layer_2],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, output_layer_2],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")


        self.trainable_variables = [self.W1, self.B1, self.W2, self.B2]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

    # this reshape "flattens" the image data
        inputs = np.reshape(inputs, [inputs.shape[0],-1])
        x1 = linear_unit(inputs, self.W1, self.B1)
        inputs = tf.nn.relu(x1)
        x = linear_unit(inputs, self.W2, self.B2)
        return x

class ModelPart2:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input_layer_1 = 8 * 8 * 64
        output_layer_1 = 256
        input_layer_2 = 256
        output_layer_2 = 2

        self.filters1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, 32],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters1")

        self.filters2 = tf.Variable(tf.random.truncated_normal([3, 3, 32, 64],
          dtype=tf.float32,
          stddev=1e-1),
          name="filters2")

        self.W1 = tf.Variable(tf.random.truncated_normal([input_layer_1, output_layer_1],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output_layer_1],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")
        self.W2 = tf.Variable(tf.random.truncated_normal([input_layer_2, output_layer_2],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, output_layer_2],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")


        self.trainable_variables = [self.filters1, self.filters2, self.W1, self.B1, self.W2, self.B2]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

        # Convolutional layer one
        x0 = tf.nn.conv2d(inputs, self.filters1, strides=[1,1,1,1], padding="SAME")
        inputs = tf.nn.relu(x0)
        inputs = tf.nn.max_pool2d(inputs, ksize=[2, 2], strides=[2,2], padding="SAME")

        # Convolutional layer two
        x1 = tf.nn.conv2d(inputs, self.filters2, strides=[1,1,1,1], padding="SAME")
        inputs = tf.nn.relu(x1)
        inputs = tf.nn.max_pool2d(inputs, ksize=[2, 2], strides=[2,2], padding="SAME")

        # Reshape inputs for linear layers
        inputs = tf.reshape(inputs, [inputs.shape[0],-1])

        # Linear layers
        x2 = linear_unit(inputs, self.W1, self.B1)
        inputs = tf.nn.relu(x2)
        x = linear_unit(inputs, self.W2, self.B2)
        return x

def loss(logits, labels):
  """
  Calculates the cross-entropy loss after one forward pass.
  :param logits: during training, a matrix of shape (batch_size, self.num_classes)
  containing the result of multiple convolution and feed forward layers
  Softmax is applied in this function.
  :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
  :return: the loss of the model as a Tensor
  """

  batch_size = len(labels)
  loss_vals = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
  average_loss = tf.reduce_mean(loss_vals)

  return average_loss

def accuracy(logits, labels):
  """
  Calculates the model's prediction accuracy by comparing
  logits to correct labels – no need to modify this.
  :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
  containing the result of multiple convolution and feed forward layers
  :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

  NOTE: DO NOT EDIT

  :return: the accuracy of the model as a Tensor
  """
  correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
  '''
  Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
  and labels - ensure that they are shuffled in the same order using tf.gather.
  You should batch your inputs.
  :param model: the initialized model to use for the forward pass and backward pass
  :param train_inputs: train inputs (all inputs to use for training),
  shape (num_inputs, width, height, num_channels)
  :param train_labels: train labels (all labels to use for training),
  shape (num_labels, num_classes)
  :return: None
  '''
  # Shuffle inputs and labels to diversify batches each training round
  indices = np.arange(len(train_labels))
  indices = tf.random.shuffle(indices)
  train_inputs = tf.gather(train_inputs, indices)
  train_labels = tf.gather(train_labels, indices)

  # Puts the data into batches 
  for start in range(0, len(train_inputs), model.batch_size):
    inputs = train_inputs[start:start+model.batch_size]
    labels = train_labels[start:start+model.batch_size]

    # For every batch, train the data
    with tf.GradientTape() as tape:
      logits = model.call(inputs) 
      average_loss = loss(logits, labels)

    gradients = tape.gradient(average_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
  """
  Tests the model on the test inputs and labels.
  :param test_inputs: test data (all images to be tested),
  shape (num_inputs, width, height, num_channels)
  :param test_labels: test labels (all corresponding labels),
  shape (num_labels, num_classes)
  :return: test accuracy - this can be the average accuracy across
  all batches or the sum as long as you eventually divide it by batch_size
  """
  return accuracy(model.call(test_inputs), test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
  """
  Uses Matplotlib to visualize the results of our model.
  :param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
  :param probabilities: the output of model.call(), shape (10, num_classes)
  :param image_labels: the labels from get_data(), shape (10, num_classes)
  :param first_label: the name of the first class, "dog"
  :param second_label: the name of the second class, "cat"

  NOTE: DO NOT EDIT

  :return: doesn't return anything, a plot should pop-up
  """
  predicted_labels = np.argmax(probabilities, axis=1)
  num_images = image_inputs.shape[0]

  fig, axs = plt.subplots(ncols=num_images)
  fig.suptitle("PL = Predicted Label\nAL = Actual Label")
  for ind, ax in enumerate(axs):
      ax.imshow(image_inputs[ind], cmap="Greys")
      pl = first_label if predicted_labels[ind] == 0.0 else second_label
      al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
      ax.set(title="PL: {}\nAL: {}".format(pl, al))
      plt.setp(ax.get_xticklabels(), visible=False)
      plt.setp(ax.get_yticklabels(), visible=False)
      ax.tick_params(axis='both', which='both', length=0)
  plt.show()


CLASS_CAT = 3
CLASS_DOG = 5
EPOCHS = 25
def main(cifar10_data_folder):
  '''
  Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
  test your model for a number of epochs. We recommend that you train for
  25 epochs.
  You should receive a final accuracy on the testing examples for cat and dog
  of ~60% for Part1 and ~70% for Part3.
  :return: None
  '''

  train_file_path = cifar10_data_folder + '/train'
  test_file_path = cifar10_data_folder + '/test'

  # Get and preprocess the data
  train_inputs, train_labels = get_data(train_file_path, CLASS_CAT, CLASS_DOG)
  test_inputs, test_labels = get_data(test_file_path, CLASS_CAT, CLASS_DOG)
  print("Data has been preprocessed")

  # Initialise the model
  model = ModelPart2()

  # Train for multiple epochs, shuffling inputs and labels each time
  for i in range(EPOCHS):
    print("Training Epoch", i)
    train(model, train_inputs, train_labels)

  # Test the network
  test_accuracy = test(model, test_inputs, test_labels)
  print("ACCURACY:", test_accuracy)
  visualize_results(test_inputs[0:10], model.call(test_inputs[0:10]), test_labels[0:10], "dog", "cat")


if __name__ == '__main__':
    local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    cifar_data_folder = local_home + '/CIFAR_data/'
    main(cifar_data_folder)
