import time
import logging
from functools import wraps
logger = logging.getLogger('cadCAD')
logger.setLevel(logging.DEBUG)
from pprint import pp

def print_time(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
      # Current timestep
      # t = len(args[2])
      t1 = time()
      f_out = f(*args, **kwargs)
      t2 = time()
      text = f"|{f.__name__} output (exec time: {t2 - t1:.2f}s)"
      logging.debug(text)
      return f_out
  return wrapper
