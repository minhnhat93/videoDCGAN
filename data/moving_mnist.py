from tensorflow.contrib.learn.python.learn.datasets.mnist import base, dtypes
from tensorflow.python.framework import random_seed
import numpy as np
from os.path import join


class DataSet(object):

  def __init__(self,
               images,
               dtype=dtypes.float32,
               seed=None,
               rescale=False,
               num_channels=1):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[-1, 1] (FOR GAN training with tanh output unit)`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
      if rescale:
        images = images * 2.0 - 1  # RESCALE IMAGE INTO [-1, 1] range for GANs
      if num_channels == 3:
        images = np.repeat(images, 3, axis=-1)
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end]


def read_data_sets(train_dir,
                   dtype=dtypes.float32,
                   train_size=10000,
                   validation_size=0,
                   seed=None,
                   rescale=False,
                   num_channels=3):

  DATA_PATH = join(train_dir, 'mnist_test_seq.npy')
  data = np.load(DATA_PATH)
  data = np.transpose(data, [1, 0, 2, 3])
  data = np.expand_dims(data, axis=-1)
  train_videos = data[:train_size]
  test_videos = data[train_size:]

  if not 0 <= validation_size <= train_size:
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(train_size, validation_size))

  validation_videos = train_videos[:validation_size]
  train_videos = train_videos[validation_size:]

  train = DataSet(
      train_videos, dtype=dtype, seed=seed, rescale=rescale, num_channels=num_channels)
  validation = DataSet(
      validation_videos,
      dtype=dtype,
      seed=seed, rescale=rescale, num_channels=num_channels)
  test = DataSet(
      test_videos, dtype=dtype, seed=seed, rescale=rescale, num_channels=num_channels)

  return base.Datasets(train=train, validation=validation, test=test)


def load_moving_mnist(train_dir='data/Moving_MNIST', rescale=True, num_channels=3):
  return read_data_sets(train_dir, rescale=rescale, num_channels=num_channels)