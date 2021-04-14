from time import time
import logging
from functools import wraps
logging.basicConfig(level=logging.DEBUG)

def print_time(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
      # Current timestep
      t = len(args[2])
      t1 = time()
      f_out = f(*args, **kwargs)
      t2 = time()
      text = f"{t}|{f.__name__} output (exec time: {t2 - t1:.2f}s)"
      logging.debug(text)
      return f_out
  return wrapper

def get_observed_initial_conditions(initial_conditions, observers):
    # Add initial observed values
    for k, f in observers.items():
        initial_conditions[k] = f({}, 0, [], initial_conditions, {})[1]
    return initial_conditions

def get_observed_psubs(psubs, observers):
    # Add observers to the variable updates in psubs
    for psub in psubs:
        for k, f in observers.items():
            psub["variables"][k] = f
    return psubs

def get_observed_params(params, observers):
    for k, f in observers.items():
        params[k] = []
    return params

def loop_time(params, step, sL, s, _input):
    if s["timestep"] % 100 == 0:
        print(s["timestep"], ">> loop_duration =", s["loop_duration"], ", total time =", s["loop_cum"])
    return ("loop_time", time())

def loop_duration(params, step, sL, s, _input):
    return ("loop_duration", time() - s["loop_time"])

def loop_cum(params, step, sL, s, _input):
    return ("loop_cum", s["loop_cum"] + s["loop_duration"])

def add_loop_ic(initial_conditions):
    initial_conditions["loop_time"] = time()
    initial_conditions["loop_duration"] = 0
    initial_conditions["loop_cum"] = 0
    return initial_conditions

def add_loop_psubs(psubs):
    psubs[-1]["variables"]["loop_duration"] = loop_duration
    psubs[-1]["variables"]["loop_cum"] = loop_cum
    psubs[-1]["variables"]["loop_time"] = loop_time
    return psubs

def add_loop_params(params):
    params["loop_time"] = []
    params["loop_duration"] = []
    params["loop_cum"] = []
    return params
