import json
import os
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from generator import Generator
from data.moving_mnist import read_data_sets
from discriminator import Discriminator
from utils import save_gif_from_sampled_videos, restore_vars, add_gradient_summaries, mkdir, \
  cleanup_dir, add_variable_summaries

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("loss_name", "WGAN-GP", "loss name: ALTERNATIVE, BASIC, WGAN-GP")
flags.DEFINE_float("gradient_policy_scale", 200.0, "labmda of gradient policy")
flags.DEFINE_integer("train_size", 10000, "size of train split")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("total_batches", -1, "amount of batch run")
flags.DEFINE_integer("hidden_dim", 128, "hidden dim of LSTM")
flags.DEFINE_integer("sequence_length", 20, "number of frames in training")
flags.DEFINE_integer("num_channels", 1, "1 = grayscale, 3 = RGB...")
flags.DEFINE_integer("learning_rate_decay_schedule", 0, "number of iters for lr decay. put 0 to not decay")
flags.DEFINE_float("learning_rate_decay_rate", 0.96, "lr decay rate")
flags.DEFINE_float("d_lr", 0.0002, "lr of critic")
flags.DEFINE_float("adam_b1", 0.5, "adam b1")
flags.DEFINE_float("adam_b2", 0.9, "adam b2")
flags.DEFINE_float("g_lr_lstm", 0.0001, "lr of generator lstm")
flags.DEFINE_float("g_lr_conv", 0.0002, "lr of generator conv")
flags.DEFINE_float("g_momentum_lstm", 0.99, "momentum for SGD of generator LSTM")
flags.DEFINE_float("regularizer_scale", 0.0, "L2 regularizer, for critic only")
flags.DEFINE_integer("sample_every_n_batches", 200, "save gif")
flags.DEFINE_integer("save_every_n_batches", 200, "checkpoint iter")
flags.DEFINE_float("dropout_kept_prob", 0.8, "dropout")
flags.DEFINE_bool("d_use_batch_norm", False, "")
flags.DEFINE_bool("d_use_layer_norm", False, "")  # Don't set this to True, currently has bugs
flags.DEFINE_bool("g_recurrent_use_layer_norm", True, "use layer norm in LSTM")
flags.DEFINE_bool("g_use_batch_norm", False, "")
flags.DEFINE_bool("g_use_layer_norm", True, "")
flags.DEFINE_bool("g_add_input_to_lstm", True, "add generated image to state of LSTM")  # this is working now due to layer norm scale h*U and x*W to the same scale
flags.DEFINE_string("g_nonlinearity", "relu", "generator conv nonlinearity: relu or lrelu")
flags.DEFINE_integer("d_steps", 5, "")
flags.DEFINE_integer("g_steps", 1, "")
flags.DEFINE_float("d_threshold", float("inf"), "threshold to train d loss to under")
flags.DEFINE_float("g_threshold", float("inf"), "threshold to train g loss to under")
flags.DEFINE_bool("d_gradient_clip", True, "variable gradient clipping for d")
flags.DEFINE_float("d_clip_norm", 5.0, "norm for each var")
flags.DEFINE_bool("g_gradient_clip_conv", True, "variable gradient clipping for convolution of g")
flags.DEFINE_float("g_clip_norm_conv", 5.0, "norm for each var")
flags.DEFINE_bool("g_gradient_clip_lstm", True, "variable gradient clipping for lstm of g")
flags.DEFINE_float("g_clip_norm_lstm", 5.0, "norm for each var")
flags.DEFINE_bool("cleanup", False, "delete checkpoint and summary director")
flags.DEFINE_string("ckpt_dir", "checkpoints", "checkpoint dir to load/save")
flags.DEFINE_string("ckpt_fn", "", "name of file in checkpoint dir to load from. example: model-10000")
flags.DEFINE_string("log_dir", "logs", "dir to write summaries to")
flags.DEFINE_string("out_dir", "out", "dir to save generated gif files to")
flags.DEFINE_integer("random_seed", 1337, "random seed")


def main(_):
  print("CONFIG: ")
  pprint(FLAGS.__flags)
  np.random.seed(FLAGS.random_seed)
  tf.reset_default_graph()
  tf.set_random_seed(FLAGS.random_seed)
  save_dir = os.path.join(FLAGS.ckpt_dir)

  generator = Generator(**FLAGS.__flags)
  discriminator = Discriminator(**FLAGS.__flags)

  global_step = tf.Variable(0, name="global_step", trainable=False)
  noises = generator.get_noise_tensor()
  x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_length,
                                              64, 64, FLAGS.num_channels])
  is_training = tf.placeholder(tf.bool, shape=())
  dropout_kept_prob = tf.placeholder(tf.float32, shape=())
  # x_noise = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_length, 64, 64,
  #                                             FLAGS.num_channels])
  # x_fake_noise = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_length, 64, 64,
  #                                                  FLAGS.num_channels])

  x_fake = generator.g_output_unit(noises, is_training)
  d_real, d_real_logit = discriminator.d_output_unit(x,
                                                     dropout_kept_prob, is_training)
  d_fake, d_fake_logit = discriminator.d_output_unit(x_fake, dropout_kept_prob,
                                                     is_training)
  if FLAGS.loss_name == "BASIC":
    g_loss = -tf.reduce_mean(tf.log(d_fake))
    d_loss_real = -tf.reduce_mean(tf.log(d_real))
    d_loss_fake = -tf.reduce_mean(tf.log(1. - d_fake))
    d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
  elif FLAGS.loss_name == "ALTERNATIVE":
    g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.ones_like(d_fake_logit)))
    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logit,
                                              labels=tf.one_hot(tf.ones(shape=[FLAGS.batch_size], dtype=tf.uint8),
                                                                depth=2) *
                                                     tf.random_uniform([FLAGS.batch_size, 1], 0.7, 1.0)))
    d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit,
                                              labels=tf.one_hot(tf.zeros(shape=[FLAGS.batch_size], dtype=tf.uint8),
                                                                depth=2) *
                                                     tf.random_uniform([FLAGS.batch_size, 1], 0.7, 1.0)))
    d_loss = d_loss_real + d_loss_fake
  elif FLAGS.loss_name == "WGAN-GP":
    g_loss = -tf.reduce_mean(d_fake_logit)
    d_loss_real = tf.reduce_mean(d_real_logit)
    d_loss_fake = tf.reduce_mean(d_fake_logit)
    d_loss = tf.reduce_mean(d_fake_logit) - tf.reduce_mean(d_real_logit)
    epsilon = tf.random_uniform([FLAGS.batch_size, 1, 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_fake
    d_hat, d_hat_logit = discriminator.d_output_unit(x_hat, dropout_kept_prob, is_training)

    ddx = tf.gradients(d_hat_logit, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), reduction_indices=[1, 2, 3, 4]))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0) * FLAGS.gradient_policy_scale)

    d_loss = d_loss + ddx

  if FLAGS.regularizer_scale > 0:
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.regularizer_scale)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, discriminator.get_weights)
    d_loss += reg_term

  if FLAGS.loss_name == "WGAN-GP":
    d_batch_norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")
    if d_batch_norm_update_ops:
      d_loss = control_flow_ops.with_dependencies(d_batch_norm_update_ops, d_loss)
    g_batch_norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="while/generator")
    if g_batch_norm_update_ops:
      g_loss = control_flow_ops.with_dependencies(g_batch_norm_update_ops, g_loss)

  d_loss_summary_op = tf.summary.scalar("d_loss/total", d_loss)
  d_loss_real_summary_op = tf.summary.scalar("d_loss/real", d_loss_real)
  d_loss_fake_summary_op = tf.summary.scalar("d_loss/fake", d_loss_fake)
  d_loss_summary_op = tf.summary.merge([d_loss_real_summary_op, d_loss_fake_summary_op, d_loss_summary_op])
  if FLAGS.loss_name == "WGAN-GP":
    ddx_summary_op = tf.summary.scalar("d_loss/ddx", ddx)
    d_loss_summary_op = tf.summary.merge([d_loss_summary_op, ddx_summary_op])
  g_loss_summary_op = tf.summary.scalar("g_loss", g_loss)
  add_variable_summaries(tf.identity(d_real_logit, name="d_activation/real"))
  add_variable_summaries(tf.identity(d_fake_logit, name="d_activation/fake"))

  increase_global_step = tf.assign(global_step, global_step + 1)
  if FLAGS.learning_rate_decay_schedule != 0:
    d_lr = tf.train.exponential_decay(FLAGS.d_lr, global_step, FLAGS.learning_rate_decay_schedule,
                                      FLAGS.learning_rate_decay_rate, staircase=True)
    g_lr_lstm = tf.train.exponential_decay(FLAGS.g_lr_lstm, global_step, FLAGS.learning_rate_decay_schedule,
                                           FLAGS.learning_rate_decay_rate, staircase=True)
    g_lr_conv = tf.train.exponential_decay(FLAGS.g_lr_conv, global_step, FLAGS.learning_rate_decay_schedule,
                                           FLAGS.learning_rate_decay_rate, staircase=True)
  else:
    d_lr = FLAGS.d_lr
    g_lr_lstm = FLAGS.g_lr_lstm
    g_lr_conv = FLAGS.g_lr_conv

  d_opt = tf.train.AdamOptimizer(d_lr, beta1=FLAGS.adam_b1, beta2=FLAGS.adam_b2)
  g_opt_lstm = tf.train.MomentumOptimizer(g_lr_lstm, momentum=FLAGS.g_momentum_lstm)
  #g_opt_lstm = tf.train.AdamOptimizer(g_lr_lstm, beta1=FLAGS.adam_b1, beta2=FLAGS.adam_b2)
  g_opt_conv = tf.train.AdamOptimizer(g_lr_conv, beta1=FLAGS.adam_b1, beta2=FLAGS.adam_b2)

  d_gvs = d_opt.compute_gradients(d_loss, var_list=list(discriminator.get_params()))
  g_gvs_lstm = g_opt_lstm.compute_gradients(g_loss, var_list=list(generator.get_lstm_params()))
  g_gvs_conv = g_opt_conv.compute_gradients(g_loss, var_list=list(generator.get_conv_params()))

  # clip by individual norm
  if FLAGS.d_gradient_clip:
    d_gvs = [(tf.clip_by_norm(grad, clip_norm=FLAGS.d_clip_norm), var) for grad, var in d_gvs]
  if FLAGS.g_gradient_clip_lstm:
    #gradients, variables = list(zip(*g_gvs_lstm))
    #gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=FLAGS.g_clip_norm_lstm)
    #g_gvs_lstm = [(grad, var) for grad, var in zip(gradients, variables)]
    g_gvs_lstm = [(tf.clip_by_norm(grad, clip_norm=FLAGS.g_clip_norm_lstm), var) for grad, var in g_gvs_lstm]
  if FLAGS.g_gradient_clip_conv:
    g_gvs_conv = [(tf.clip_by_norm(grad, clip_norm=FLAGS.g_clip_norm_conv), var) for grad, var in g_gvs_conv]

  add_gradient_summaries(d_gvs)
  add_gradient_summaries(g_gvs_lstm)
  add_gradient_summaries(g_gvs_conv)

  d_solver = d_opt.apply_gradients(d_gvs)
  g_solver_lstm = g_opt_lstm.apply_gradients(g_gvs_lstm)
  g_solver_conv = g_opt_conv.apply_gradients(g_gvs_conv)

  if FLAGS.loss_name in ["BASIC"]:
    moving_mnist = read_data_sets("data/Moving_MNIST", train_size=FLAGS.train_size, rescale=False, num_channels=FLAGS.num_channels)
  else:
    moving_mnist = read_data_sets("data/Moving_MNIST", train_size=FLAGS.train_size, rescale=True, num_channels=FLAGS.num_channels)

  mkdir(FLAGS.out_dir)
  mkdir(FLAGS.log_dir)
  if FLAGS.cleanup and FLAGS.ckpt_fn is None:
    cleanup_dir(FLAGS.out_dir)
    cleanup_dir(FLAGS.log_dir)

  discriminator.create_summaries_for_variables()
  generator.create_summaries_for_variables()
  summary_op = tf.summary.merge_all()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=sess.graph)

  restore_vars(saver, sess, save_dir, FLAGS.ckpt_fn)
  iter = int(global_step.eval(sess))
  start_iter = iter
  json.dump(FLAGS.__flags, open(os.path.join(save_dir, "config.txt"), "w"))

  summary_noises = generator.generate_noise()
  num_iters_pretrain_d = int(25.0 * FLAGS.batch_size / FLAGS.train_size)

  while True:
    if iter > start_iter and iter % FLAGS.sample_every_n_batches == 0:
      videos = sess.run(x_fake, feed_dict={noises: summary_noises, is_training: True})
      save_gif_from_sampled_videos(videos, fn="{}/{}.gif".format(FLAGS.out_dir, str(iter).zfill(7)))

    if iter > start_iter and iter % FLAGS.save_every_n_batches == 0:
      print("Saving model begins...")
      saver.save(sess, os.path.join(save_dir, "model"), global_step=iter)
      print("Save completed!")

    real_videos = moving_mnist.train.next_batch(FLAGS.batch_size)

    if iter < num_iters_pretrain_d or iter % 500 == 0:
      d_iters = 100
    else:
      d_iters = FLAGS.d_steps
    g_iters = FLAGS.g_steps
    for j in range(d_iters):
      _, d_loss_curr, d_loss_summary = sess.run([d_solver, d_loss, d_loss_summary_op],
                                                feed_dict={noises: generator.generate_noise(),
                                                           x: real_videos,
                                                           dropout_kept_prob: FLAGS.dropout_kept_prob,
                                                           is_training: True})
      if d_loss_curr < FLAGS.d_threshold:
        break

    for j in range(g_iters):
      _, _, g_loss_curr, g_loss_summary = sess.run([g_solver_lstm, g_solver_conv, g_loss, g_loss_summary_op],
                                                feed_dict={noises: generator.generate_noise(),
                                                           dropout_kept_prob: FLAGS.dropout_kept_prob,
                                                           is_training: True})
      if g_loss_curr < FLAGS.g_threshold:
        break

    print("Step {}: D loss {:.6f}, G loss {:.6f}".format(iter, d_loss_curr, g_loss_curr))

    if iter > start_iter and iter % FLAGS.save_every_n_batches == 0:
      summary = sess.run(summary_op, feed_dict={noises: summary_noises,
                                                x: real_videos,
                                                dropout_kept_prob: FLAGS.dropout_kept_prob,
                                                is_training: True})
      writer.add_summary(summary, global_step=iter)
      writer.flush()
    else:
      writer.add_summary(d_loss_summary, global_step=iter)
      writer.add_summary(g_loss_summary, global_step=iter)
    sess.run([increase_global_step])
    iter += 1
    if iter == FLAGS.total_batches:
      break

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
