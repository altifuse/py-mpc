# python libs
import numpy as np
import time
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

global cymodels


def initialize_models():
    global cymodels
    from mpc import cymodels


def prediction(models, actuators, history_array, prediction_horizon, history_length, initial_tick):
    predictions = [list() for i in range(uwabami.Tick.number_of_models)]

    for t in range(prediction_horizon):
        new_tick = history_array[-uwabami.Tick.number_of_columns:].copy()
        history_array = np.concatenate([[0], history_array])
        for i in range(uwabami.Tick.number_of_models):
            new_tick[i] = cymodels.eval(i, history_array)
            predictions[i].append(new_tick[i])
        # replacing actuators:
        new_tick[uwabami.Tick.steer_index], \
            new_tick[uwabami.Tick.accel_index], \
            new_tick[uwabami.Tick.brake_index] = actuators[((uwabami.Tick.number_of_actuators - 1) * t):(
                (uwabami.Tick.number_of_actuators - 1) * (t + 1))]

        # update history with current tick and remove oldest tick to enforce history length
        history_array = np.concatenate([history_array[uwabami.Tick.number_of_columns:], new_tick])
    return predictions


def prepare_tick_history(tick_history):
    new_tick_history = tick_history[0].get_numpy_array()
    for tick in tick_history[1:]:
        new_tick_history = np.append(new_tick_history, tick.get_numpy_array())
    return new_tick_history
