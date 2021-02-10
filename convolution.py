from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
  """
  Performs 2D convolution given 4D inputs and filter Tensors.
  :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
  :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
  :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
  :param padding: either "SAME" or "VALID", capitalization matters
  :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
  """
  num_examples = tf.shape(inputs)[0]
  in_height = tf.shape(inputs)[1]
  in_width = tf.shape(inputs)[2]
  input_in_channels = tf.shape(inputs)[3]

  filter_height = tf.shape(filters)[0]
  filter_width = tf.shape(filters)[1]
  filter_in_channels = tf.shape(filters)[2]
  filter_out_channels = tf.shape(filters)[3]

  assert filter_in_channels != input_in_channels, "ERROR: FILTER AND INPUT IN CHANNELS DO NOT MATCH"

  num_examples_stride = strides[0]
  stride_y = strides[1]
  stride_x = strides[2]
  channels_stride = strides[3]

  # Padding input
  if padding == "SAME":
    pad_y = tf.cast(tf.math.floor((filter_height - 1) / 2), tf.int32)
    pad_x = tf.cast(tf.math.floor((filter_width - 1) / 2), tf.int32)
    num_examples_pad = [0,0]
    height_pad = [pad_y, pad_y]
    width_pad = [pad_x, pad_x]
    channels_pad = [0, 0]
    inputs = np.pad(inputs, (num_examples_pad, height_pad, width_pad, channels_pad))
  elif padding != "VALID":
    print("ERROR: PADDING ARGUMENT MUST BE 'SAME' OR 'VALID'")

  inputs = tf.expand_dims(inputs, axis=4)
  inputs = tf.repeat(inputs, filter_out_channels, axis=4)

  # Calculate output dimensions
  output_height = tf.cast((tf.shape(inputs)[1] - filter_height + 1), tf.int32)
  output_width = tf.cast((tf.shape(inputs)[2] - filter_width + 1), tf.int32)
  output_channels = filter_out_channels
  output_shape = (num_examples, output_height, output_width, output_channels)
  output = tf.Variable(tf.zeros(output_shape, dtype=tf.float32))

  # Carry out convolution
  for i in range(in_height-filter_height + 1):
    for j in range(in_width-filter_width + 1):
      axes = [1, 2, 3]
      input_beneath_filter = inputs[:, i:(i+filter_height), j:(j+filter_width), :]
      multiplied = tf.math.multiply(input_beneath_filter, filters)
      reduced = tf.reduce_sum(multiplied, axes)
      output[:, i, j, :].assign(reduced)
