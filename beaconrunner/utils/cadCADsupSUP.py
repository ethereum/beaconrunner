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
      logger.debug(text)
      return f_out
  return wrapper

def get_observed_initial_conditions(initial_conditions, observers):
    # Add initial observed values
    for k, f in observers.items():
        initial_conditions[k] = f({}, 0, [], initial_conditions, {})[1]
    initial_conditions["run_time"] = 0
    return initial_conditions

def update_run_time(params, substep, state_history, previous_state, policy_input):
    return ('run_time', time.time())

def get_observed_psubs(psubs, observers):
    # Add observers to the variable updates in psubs
    for psub in psubs:
        for k, f in observers.items():
            psub["variables"][k] = f

    measure_run_time_block = {
        'policies': {},
        'variables': {
            'run_time': update_run_time
        }
    }

    profiled_state_update_blocks = []
    profiled_state_update_blocks.append({
        'label': 'Initial measure runtime',
        'policies': {},
        'variables': {
            'run_time': update_run_time
        },
    })

    for index, block in enumerate(psubs):
        profiled_state_update_blocks.append(block)
        profiled_state_update_blocks.append({
            'label': f'Measure runtime {index}',
            'policies': {},
            'variables': {
                'run_time': update_run_time
            },
        })

    return profiled_state_update_blocks

def get_observed_params(params, observers):
    for k, f in observers.items():
        params[k] = []
    return params

def loop_time(params, step, sL, s, _input):
    print(s["timestep"], ">> loop_duration =", s["loop_duration"], ", total time =", s["loop_cum"])
    return ("loop_time", time())

def loop_duration(params, step, sL, s, _input):
    return ("loop_duration", time() - s["loop_time"])

def loop_cum(params, step, sL, s, _input):
    return ("loop_cum", s["loop_cum"] + s["loop_duration"])

def add_loop_ic(initial_conditions):
    initial_conditions["loop_time"] = time.time()
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
