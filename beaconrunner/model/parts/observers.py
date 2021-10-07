import time

# Observers core

def update_run_time(params, substep, state_history, previous_state, policy_input):
    return ('run_time', time.time())

def get_observed_initial_conditions(initial_conditions, observers):
    # Add initial observed values
    for k, f in observers.items():
        initial_conditions[k] = f({}, 0, [], initial_conditions, {})[1]
    initial_conditions["run_time"] = 0
    return initial_conditions

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

def add_observers(initial_conditions, psubs, observers):
    observed_ic = get_observed_initial_conditions(initial_conditions, observers)
    observed_psubs = get_observed_psubs(psubs, observers)
    return (observed_ic, observed_psubs)
