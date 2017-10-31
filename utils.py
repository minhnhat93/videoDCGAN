import os

import matplotlib
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageChops, ImageOps
import shutil


def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def cleanup_dir(dir_name):
  if os.path.exists(dir_name):
    shutil.rmtree(dir_name)
  mkdir(dir_name)


def generate_samples(sess, trainable_model, batch_size, generated_num):
  # Generate Samples
  generated_samples = []
  for _ in range(int(generated_num / batch_size)):
    generated_samples.extend(trainable_model.generate(sess))

  return generated_samples


def plot_samples(samples, nrows=4, ncols=4, vmin=-1.0, vmax=1.0):
  # sample of shape: num_sample x 64 x 64 x 3
  sample_w, sample_h = samples.shape[1:3]
  DPI = 100
  border_width = 2
  width = ((sample_w + 2 * border_width) * nrows + 2 * border_width)
  height = ((sample_h + 2 * border_width) * ncols + 2 * border_width)
  fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
  fig.patch.set_facecolor('white')

  pixel_width = 1.0 / width
  pixel_height = 1.0 / height
  for i, sample in enumerate(samples):
    if i >= nrows * ncols:
      break
    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_aspect('equal')
    _x = float(i // ncols)
    _y = float(i % ncols)
    x = pixel_width * (_x * sample_w + (2 * _x + 1) * border_width)
    y = pixel_height * (_y * sample_h + (2 * _y + 1) * border_width)
    ax.set_position([x, y, sample_w * pixel_width, sample_h * pixel_height])
    sample = sample.squeeze()
    if len(sample.shape) == 2:
      plt.imshow(sample, cmap="gray", vmin=vmin, vmax=vmax)
    else:
      plt.imshow(sample, vmin=vmin, vmax=vmax)
  # fig.subplots_adjust(wspace=0.2 / ncols, hspace=0.1 / nrows)
  # fig.subplots_adjust(wspace=0, hspace=0)
  width, height = (fig.get_size_inches() * fig.get_dpi()).astype(np.int32)
  canvas = fig.canvas
  canvas.draw()
  frame = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape((width, height, 3))
  plt.close(fig)
  del fig
  return frame, width, height


def save_gif(filename, array, fps=10, scale=1.0):
  """Creates a gif given a stack of images using moviepy
  Notes
  -----
  works with current Github version of moviepy (not the pip version)
  https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
  CREDIT. Got it from: https://gist.github.com/nirum/d4224ad3cd0d71bfef6eba8f3d6ffd59
  Usage
  -----
  >>> X = randn(100, 64, 64)
  >>> gif('test.gif', X)
  Parameters
  ----------
  filename : string
      The filename of the gif to write to
  array : array_like
      A numpy array that contains a sequence of images
  fps : int
      frames per second (default: 10)
  scale : float
      how much to rescale each image by (default: 1.0)
  """

  # ensure that the file has the .gif extension
  fname, _ = os.path.splitext(filename)
  filename = fname + '.gif'

  # copy into the color dimension if the images are black and white
  if array.ndim == 3:
    array = array[..., np.newaxis] * np.ones(3)

  # make the moviepy clip
  clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
  clip.write_gif(filename, fps=fps)
  return clip


def trim_im(im):
  bg = Image.new(im.mode, im.size, im.getpixel((im.size[0] - 1, im.size[1] - 1)))
  diff = ImageChops.difference(im, bg)
  diff = ImageChops.add(diff, diff, 2.0, -100)
  bbox = diff.getbbox()
  if bbox:
    return im.crop(bbox)


def release_list(a):
  del a[:]
  del a


def save_gif_from_sampled_videos(videos, fn, ncols=4, nrows=4):
  # videos is of shape: num_sample x num_frame x 64 x 64 x 3
  frames = []
  videos = videos.transpose((1, 0, 2, 3, 4))
  for samples in videos:
    frame, width, height = plot_samples(samples, ncols=ncols, nrows=nrows)
    # frame = trim_im(Image.fromarray(frame))
    # frame = Image.fromarray((frame))
    # frame = ImageOps.expand(frame, border=5, fill='white')
    frame = np.asarray(frame)
    frames.append(frame)
  frames = np.asarray(frames)
  save_gif(fn, frames)


def restore_vars(saver, sess, chkpt_dir, fn):
  """ Restore saved net, global score and step, and epsilons OR
  create checkpoint directory for later storage. """
  if not os.path.exists(chkpt_dir):
    try:
      os.makedirs(chkpt_dir)
    except OSError:
      pass
  if not fn:
    path = chkpt_dir
    print("Saving at:", path)
  else:
    path = os.path.join(chkpt_dir, str(fn))
    saver.restore(sess, path)
    print("Loading at:", path)


def add_variable_summaries(var, var_name=None):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  if var_name is None:
    var_name = var.op.name
  with tf.name_scope('{}/summaries'.format(var_name)):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_gradient_summaries(grads_and_vars):
  for grad, var in grads_and_vars:
    if grad is not None:
      tf.summary.histogram(var.op.name + "/gradient", grad)


def add_activation_summary(var):
  tf.summary.histogram(var.op.name + "/activation", var)
  tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def create_string_from_config(config):
  out_str = ""
  for key in sorted(config.keys()):
    if key in ["ckpt_dir", "ckpt_fn"]:
      continue
    out_str += str(config[key]) + '_'
  out_str = out_str[:-1]
  out_str = ''.join(c for c in out_str if c.isalnum() or c in ['_'])
  if len(out_str) > 256:
    out_str = out_str[:256]
  return out_str
