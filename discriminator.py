import tensorflow as tf

from ops import *
from utils import add_activation_summary
from tfnet import TFNet


class Discriminator(TFNet):
  def __init__(self, **kwargs):
    super(Discriminator, self).__init__()
    self.batch_size = kwargs.get("batch_size")
    self.sequence_length = kwargs.get("sequence_length")
    self.use_batch_norm = kwargs.get("d_use_batch_norm")
    self.use_layer_norm = kwargs.get("d_use_layer_norm")
    self.dropout_kept_prob = kwargs.get("dropout_kept_prob")
    self.loss_name = kwargs.get("loss_name")
    self.d_output_unit = self.create_discriminator_unit()

  def create_discriminator_unit(self):
    def unit(input_videos, dropout_kept_prob, is_training):
      if self.loss_name in ["ALTERNATIVE", "BASIC"]:
        in_place_update = True
      else:
        in_place_update = False
      with tf.variable_scope("critic"):
        # batch norm must be between conv and relu, input to discriminator no batch norm
        conv1, W_conv1, b_conv1 = conv3d(input_videos, 64, 4, 4, 4, 2, 2, 2, stddev=0.02, name="conv1",
                                                   with_w=True)
        relu1 = lrelu(conv1, name="relu1")

        conv2, W_conv2, b_conv2 = conv3d(relu1, 128, 4, 4, 4, 2, 2, 2, stddev=0.02, name="conv2", with_w=True)
        if self.use_batch_norm:
          batch_norm2 = batch_norm(conv2, is_training, in_place_update=in_place_update, name="batch_norm2")
          relu2 = lrelu(batch_norm2, name="relu2")
        elif self.use_layer_norm:
          layer_norm2, scale2, shift2 = layer_norm(conv2, with_w=True, scope="layer_norm2")
          relu2 = lrelu(layer_norm2, name='relu2')
        else:
          relu2 = lrelu(conv2, name="relu2")

        conv3, W_conv3, b_conv3 = conv3d(relu2, 256, 4, 4, 4, 2, 2, 2, stddev=0.02, name="conv3", with_w=True)
        if self.use_batch_norm:
          batch_norm3 = batch_norm(conv3, is_training, in_place_update=in_place_update, name="batch_norm3")
          relu3 = lrelu(batch_norm3, name="relu3")
        elif self.use_layer_norm:
          layer_norm3, scale3, shift3 = layer_norm(conv3, with_w=True, scope="layer_norm3")
          relu3 = lrelu(layer_norm3, name='relu3')
        else:
          relu3 = lrelu(conv3, name="relu3")

        conv4, W_conv4, b_conv4 = conv3d(relu3, 512, 4, 4, 4, 2, 2, 2, stddev=0.02, name="conv4", with_w=True)
        if self.use_batch_norm:
          batch_norm4 = batch_norm(conv4, is_training, in_place_update=in_place_update, name="batch_norm4")
          relu4 = lrelu(batch_norm4, name="relu4")
        elif self.use_layer_norm:
          layer_norm4, scale4, shift4 = layer_norm(conv4, with_w=True, scope="layer_norm4")
          relu4 = lrelu(layer_norm4, name='relu4')
        else:
          relu4 = lrelu(conv4, name="relu4")

        dropout4 = tf.nn.dropout(relu4, dropout_kept_prob, name="dropout4")

        if self.loss_name in ["BASIC", "WGAN-GP"]:
          out_dim = 1
        else:
          out_dim = 2
        d_logit, self.W_logit, self.b_logit = linear(tf.contrib.layers.flatten(dropout4), out_dim, stddev=0.02,
                                                     name="log_prob", with_biases=self.loss_name in ["BASIC", "ALTERNATIVE"],
                                                     with_w=True)
        d_prob = tf.nn.sigmoid(d_logit, name="prob")

      return d_prob, d_logit

    return unit

  def get_params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")

  def get_weights(self):
    weights = []
    for j in range(1, 5):
      weights += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic/conv{}/w".format(j))
    weights += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic/log_prob/weight")
    return weights
