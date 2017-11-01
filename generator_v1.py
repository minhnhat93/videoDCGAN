# Implementation based on these two files
# - https://github.com/LantaoYu/SeqGAN/blob/master/generator.py
# - https://github.com/Yuliang-Zou/tf_videogan/blob/master/main.py
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

from ops import *
from tfnet import TFNet


class Generator(TFNet):
  def __init__(self, **kwargs):
    super(Generator, self).__init__()
    self.batch_size = kwargs.get("batch_size")
    self.hidden_dim = kwargs.get("hidden_dim")  # lstm hidden state dimension
    self.sequence_length = kwargs.get("sequence_length")  # sequence length
    self.num_channels = kwargs.get("num_channels")
    self.use_batch_norm = kwargs.get("g_use_batch_norm")
    self.use_layer_norm = kwargs.get("g_use_layer_norm")
    self.recurrent_use_layer_norm = kwargs.get("g_recurrent_use_layer_norm")
    self.add_input_to_lstm = kwargs.get("g_add_input_to_lstm")
    self.loss_name = kwargs.get("loss_name")
    self.input_size = 64 * 64 * self.num_channels  # image size is fixed at 64 by 64, input to LSTM is image flattened
    nonlinearity = kwargs.get("g_nonlinearity")
    if nonlinearity == "relu":
      self.nonlinearity = tf.nn.relu
    else:
      self.nonlinearity = lrelu

    self.g_output_unit = self.create_output_unit()

  def init_matrix(self, shape):
    # Glorot initialization
    # stddev = np.sqrt(6.0 / (shape[0] + shape[1]))
    # stddev = 0.01
    stddev = 0.1
    return tf.random_normal(shape, stddev=stddev)

  def init_biases(self, hidden_dim):
    # forget gate must be init to ones
    b_f = tf.ones([hidden_dim])
    b_rest = tf.zeros([3 * hidden_dim])
    b = tf.concat([b_f, b_rest], axis=0)
    return b

  def create_recurrent_unit(self, scope="generator/lstm"):
    # See official layer norm implementation at: https://github.com/ryankiros/layer-norm/blob/master/layers.py#L457
    with tf.variable_scope(scope):
      U = tf.Variable(self.init_matrix([self.hidden_dim, 4 * self.hidden_dim]), name="U")
      if self.add_input_to_lstm:
        W = tf.Variable(self.init_matrix([self.input_size, 4 * self.hidden_dim]), name="W")
      b = tf.Variable(self.init_biases(self.hidden_dim), name="b")
      if self.recurrent_use_layer_norm:
        scale_h = tf.Variable(tf.ones([4 * self.hidden_dim]), name='scale_h')
        shift_h = tf.Variable(tf.zeros([4 * self.hidden_dim]), name='shift_h')
        if self.add_input_to_lstm:
          scale_x = tf.Variable(tf.ones([4 * self.hidden_dim]), name='scale_x')
          shift_x = tf.Variable(tf.zeros([4 * self.hidden_dim]), name='shift_x')
        scale_c = tf.Variable(tf.ones([self.hidden_dim]), name='scale_c')
        shift_c = tf.Variable(tf.zeros([self.hidden_dim]), name='shift_c')

    def unit(x, hidden_memory_tm1):
      h_prev, c_prev = tf.unstack(hidden_memory_tm1)
      h_prev_ = tf.matmul(h_prev, U)
      if self.recurrent_use_layer_norm:
        h_prev_ = layer_norm(h_prev_, scale=scale_h, shift=shift_h, scope="layer_norm_h")
      this_state = h_prev_ + b
      if self.add_input_to_lstm:
        x_ = tf.matmul(x, W)
        if self.recurrent_use_layer_norm:
          x_ = layer_norm(x_, scale=scale_x, shift=shift_x, scope="layer_norm_x")
        this_state = this_state + x_
      f, i, o, c = tf.split(this_state, [self.hidden_dim] * 4, axis=1)
      f, i, o, c = tf.sigmoid(f), tf.sigmoid(i), tf.sigmoid(o), tf.tanh(c)
      # Final Memory cell
      c = f * c_prev + i * c
      if self.recurrent_use_layer_norm:
        c_ = layer_norm(c, scale=scale_c, shift=shift_c, scope="layer_norm_c")
      else:
        c_ = c
      # Current Hidden state
      h = o * tf.nn.tanh(c_)
      return tf.stack([h, c])

    return unit

  def create_conv_unit(self, scope="generator/conv"):
    def unit(hidden_memory_tuple, is_training):
      hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
      # hidden_state : batch x hidden_dim
      # expand hidden state to batch x 1 x 1 x hidden_dim
      hidden_state = tf.expand_dims(hidden_state, 1)
      hidden_state = tf.expand_dims(hidden_state, 1)
      if self.loss_name in ["ALTERNATIVE", "BASIC"]:
        in_place_update = True
      else:
        in_place_update = False

      with tf.variable_scope(scope):
        deconv1, W_deconv1, b_deconv1 = deconv2d(hidden_state, [self.batch_size, 4, 4, 512],
                                                 4, 4, 2, 2, stddev=0.02, name="deconv1", with_w=True,
                                                 padding="VALID")
        if self.use_batch_norm:
          batch_norm1 = batch_norm(deconv1, is_training, in_place_update=in_place_update, name="batch_norm1")
          relu1 = self.nonlinearity(batch_norm1, name="relu1")
        elif self.use_layer_norm:
          layer_norm1, scale1, shift1 = layer_norm(deconv1, with_w=True, scope="layer_norm1")
          relu1 = self.nonlinearity(layer_norm1, name='relu1')
        else:
          relu1 = self.nonlinearity(deconv1, name="relu1")

        deconv2, W_deconv2, b_deconv2 = deconv2d(relu1, [self.batch_size, 8, 8, 256],
                                                 4, 4, 2, 2, stddev=0.02, name="deconv2", with_w=True)
        if self.use_batch_norm:
          batch_norm2 = batch_norm(deconv2, is_training, in_place_update=in_place_update, name="batch_norm2")
          relu2 = self.nonlinearity(batch_norm2, name="relu2")
        elif self.use_layer_norm:
          layer_norm2, scale2, shift2 = layer_norm(deconv2, with_w=True, scope="layer_norm2")
          relu2 = self.nonlinearity(layer_norm2, name='relu2')
        else:
          relu2 = self.nonlinearity(deconv2, name="relu2")

        deconv3, W_deconv3, b_deconv3 = deconv2d(relu2, [self.batch_size, 16, 16, 128],
                                                 4, 4, 2, 2, stddev=0.02, name="deconv3", with_w=True)
        if self.use_batch_norm:
          batch_norm3 = batch_norm(deconv3, is_training, in_place_update=in_place_update, name="batch_norm3")
          relu3 = self.nonlinearity(batch_norm3, name="relu3")
        elif self.use_layer_norm:
          layer_norm3, scale3, shift3 = layer_norm(deconv3, with_w=True, scope="layer_norm3")
          relu3 = self.nonlinearity(layer_norm3, name='relu3')
        else:
          relu3 = self.nonlinearity(deconv3, name="relu3")

        deconv4, W_deconv4, b_deconv4 = deconv2d(relu3, [self.batch_size, 32, 32, 64],
                                                 4, 4, 2, 2, stddev=0.02, name="deconv4", with_w=True)
        if self.use_batch_norm:
          batch_norm4 = batch_norm(deconv4, is_training, in_place_update=in_place_update, name="batch_norm4")
          relu4 = self.nonlinearity(batch_norm4, name="relu4")
        elif self.use_layer_norm:
          layer_norm4, scale4, shift4 = layer_norm(deconv4, with_w=True, scope="layer_norm4")
          relu4 = self.nonlinearity(layer_norm4, name='relu4')
        else:
          relu4 = self.nonlinearity(deconv4, name="relu4")

        deconv5, W_deconv5, b_deconv5 = deconv2d(relu4, [self.batch_size, 64, 64, self.num_channels],
                                                 4, 4, 2, 2, stddev=0.02, name="deconv5", with_w=True)
        if self.loss_name in ["BASIC"]:
          x_fake = tf.nn.sigmoid(deconv5, name="x_fake")
        else:
          x_fake = tf.nn.tanh(deconv5, name="x_fake")

      return x_fake

    return unit

  def create_output_unit(self):
    g_recurrent_unit = self.create_recurrent_unit()
    g_video_generator_unit = self.create_conv_unit()

    def unit(z_0, is_training):
      # start_noise MUST have the same dimension as hidden state
      # Initial states
      # h0 = tf.zeros([self.batch_size, self.hidden_dim])
      # h0 = tf.stack([h0, h0])
      h0 = tf.stack([z_0, z_0])

      gen_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                           dynamic_size=False, infer_shape=True)

      def _g_recurrence(i, x_t, h_tm1, gen_x):
        h_t = g_recurrent_unit(tf.contrib.layers.flatten(x_t), h_tm1)  # hidden_memory_tuple
        o_t = g_video_generator_unit(h_t, is_training)  # batch x frame (64 x 64 x 3)
        x_tp1 = o_t
        gen_x = gen_x.write(i, x_tp1)  # indices, batch_size
        return i + 1, x_tp1, h_t, gen_x

      # start_input = tf.contrib.layers.flatten(g_video_generator_unit(tf.stack([start_noise, start_noise]),
      #                                                                is_training))
      x_0 = tf.zeros(shape=[self.batch_size, 64, 64, self.num_channels])
      if not self.use_batch_norm:
        # control dependencies inside while loop call not be called outside the loop. batch norm cannot set update
        # collection to None because that will cause gradient error, this is bug in tensorflow. the only way is to
        # write it explicitly during training and when testing use control_flow_ops.while_loop
        # For more information:
        # https://github.com/tensorflow/tensorflow/issues/9034#issuecomment-294619443 (batch norm inplace update has bug)
        # https://github.com/tensorflow/tensorflow/issues/6087#issuecomment-283534177 (cannot use update_ops outside)
        _, _, _, gen_x = control_flow_ops.while_loop(
          cond=lambda i, _1, _2, _3: i < self.sequence_length,
          body=_g_recurrence,
          loop_vars=(tf.constant(0, dtype=tf.int32),
                     x_0,
                     h0, gen_x))
      else:
        x_t = x_0
        h_t = h0
        for _ in range(self.sequence_length):
          _, x_t, h_t, gen_x = _g_recurrence(_, x_t, h_t, gen_x)

      gen_x = gen_x.stack()  # seq_length x batch_size x video dims
      gen_x = tf.transpose(gen_x, perm=[1, 0, 2, 3, 4])
      gen_x = tf.reshape(gen_x, [self.batch_size, self.sequence_length, 64, 64, self.num_channels])
      # output size: batch_size * sequence_length * 64 * 64 * channels
      return gen_x

    return unit

  def get_noise_tensor(self):
    return tf.placeholder(tf.float32, shape=[self.batch_size, self.hidden_dim])

  def generate_noise(self):
    # normal sampling instead of uniform: https://github.com/soumith/ganhacks
    return np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, self.hidden_dim))
    # return np.random.normal(0.0, 1.0, size=(self.batch_size, self.hidden_dim))

  def g_optimizer(self, *args, **kwargs):
    return tf.train.AdamOptimizer(*args, **kwargs)

  def get_params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

  def get_lstm_params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/lstm")

  def get_conv_params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/conv")
