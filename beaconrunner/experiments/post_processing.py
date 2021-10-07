import pandas as pd
from radcad.core import generate_parameter_sweep

from model.system_parameters import parameters, Parameters
from model.state_update_blocks import state_update_blocks

def assign_parameters(df: pd.DataFrame, parameters: Parameters, set_params=[]):
    if set_params:
        parameter_sweep = generate_parameter_sweep(parameters)
        parameter_sweep = [{param: subset[param] for param in set_params} for subset in parameter_sweep]

        for subset_index in df['subset'].unique():
            for (key, value) in parameter_sweep[subset_index].items():
                df.loc[df.eval(f'subset == {subset_index}'), key] = value

    return df


def post_process(
    df: pd.DataFrame,
    drop_timestep_zero=True,
    state_update_blocks=state_update_blocks,
    parameters=parameters,
    drop_network=False):
    # Assign parameters to DataFrame
    assign_parameters(df, parameters, [])

    # Create a substep -> label mapping
    # For each PSUB on the partial_state_update_blocks,
    # we'll retrieve the 'label' key
    # and associate it with the order on the PSUB.
    psub_map = {order + 1: psub.get('label', '')
                for (order, psub)
                in enumerate(state_update_blocks)}

    # Set substep=0 as being the inital state
    if not drop_timestep_zero:
        psub_map[0] = 'Initial State'

    # Create a new column called "substep" label
    df['substep_label'] = df.substep.map(psub_map)

    # Drop the initial state for plotting
    if drop_timestep_zero:
        df = df.drop(df.query('timestep == 0').index)

    if drop_network:
        df.drop('network', axis=1, inplace=True)

    return df
