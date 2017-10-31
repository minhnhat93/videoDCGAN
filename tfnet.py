from utils import add_variable_summaries
import tensorflow as tf


class TFNet(object):
  def __init__(self):
    self.params = None

  def get_params(self):
    raise NotImplementedError

  def create_summaries_for_variables(self):
    for var in self.get_params():
      add_variable_summaries(var)
