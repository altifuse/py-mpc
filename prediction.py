# python libs
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

global cymodels


def initialize_models():
    global cymodels
    # import cymodels # to be used with compiled models for improved performance


def prediction(models, controls, history_array, prediction_horizon, history_length, initial_tick, number_of_controls, number_of_variables):
    predictions = [list() for i in range(number_of_controls)]

    for t in range(prediction_horizon):
        new_tick = history_array[-number_of_variables:].copy()
        history_array = np.concatenate([[0], history_array])
        for i in range(number_of_controls):
            new_tick[i] = cymodels.eval(i, history_array)
            predictions[i].append(new_tick[i])

        # replacing controls:
        # new_tick[desired_indexes] = controls[((number_of_controls - 1) * t):((number_of_controls - 1) * (t + 1))]

        # update history with current tick and remove oldest tick to enforce history length
        history_array = np.concatenate([history_array[number_of_variables:], new_tick])
    return predictions


def prepare_tick_history(tick_history):
    new_tick_history = tick_history[0].get_numpy_array()
    for tick in tick_history[1:]:
        new_tick_history = np.append(new_tick_history, tick.get_numpy_array())
    return new_tick_history
