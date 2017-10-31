# Reference: https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
import tensorflow as tf
import numpy as np


def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", with_w=False, padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_dim
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv

# TODO: ADD GLOROT NORMAL INTIALIZATION
def conv3d(input_, output_dim,
           k_d=4, k_h=4, k_w=4, d_d=2, d_h=2, d_w=2, stddev=None,
           name="conv3d", with_w=False, padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_d * k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_d * k_h * k_w * output_dim
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("w", [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv


def deconv2d(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
             name="deconv2d", with_w=False, padding="SAME"):
  # Glorot initialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_shape[-1]
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    if with_w:
      return deconv, w, biases
    else:
      return deconv


def deconv3d(input_, output_shape,
             k_d=4, k_h=4, k_w=4, d_d=1, d_h=1, d_w=1, stddev=None,
             name="deconv3d", with_w=False, padding="SAME"):
  # Glorot initialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_d * k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_d * k_h * k_w * output_shape[-1]
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("w", [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_d, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name) as scope:
    return tf.maximum(x, leak * x)


def linear(input_, output_size, name="linear", stddev=None, bias_start=0.0, with_biases=True, with_w=False):
  shape = input_.get_shape().as_list()

  if stddev is None:
    stddev = np.sqrt(1. / (shape[1]))
  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                             tf.truncated_normal_initializer(stddev=stddev))
    if with_biases:
      bias = tf.get_variable("b", [output_size],
                             initializer=tf.constant_initializer(bias_start))
    if with_w:
      if with_biases:
        return tf.matmul(input_, weight) + bias, weight, bias
      else:
        return tf.matmul(input_, weight), weight, None
    else:
      if with_biases:
        return tf.matmul(input_, weight) + bias
      else:
        return tf.matmul(input_, weight)


def batch_norm(input, is_training, momentum=0.9, epsilon=1e-5, in_place_update=False, name="batch_norm"):
  if in_place_update:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        updates_collections=None,
                                        is_training=is_training,
                                        scope=name)
  else:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        is_training=is_training,
                                        scope=name)


# credit: https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
def layer_norm(tensor, scale=None, shift=None, scope=None, epsilon=1e-5, with_w=False):
  """ Layer normalizes a 2D tensor along its second axis """
  inputs_shape = tensor.get_shape()
  input_ranks = inputs_shape.ndims
  # This implementation of params_shape doesn't work well for 3D convolution. Maybe this implementation 
  # is flawed for that case. Tensorflow documentation says the implement the same way, however
  # See here: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
  # and according to the layer_norm paper
  # params_shape should be the number of convolution kernel for 2d case
  params_shape = inputs_shape.as_list()[-1:]
  params_shape = [1] * (input_ranks - 2) + params_shape
  m, v = tf.nn.moments(tensor, list(range(1, input_ranks)), keep_dims=True)
  if not isinstance(scope, str):
    scope = 'layer_norm'
  with tf.variable_scope(scope) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    if scale is None:
      scale = tf.get_variable('scale',
                              shape=params_shape,
                              initializer=tf.constant_initializer(1))
    if shift is None:
      shift = tf.get_variable('shift',
                              shape=params_shape,
                              initializer=tf.constant_initializer(0))
  LN_initial = (tensor - m) / tf.sqrt(v + epsilon)
  if with_w:
    return LN_initial * scale + shift, scale, shift
  else:
    return LN_initial * scale + shift
