# python libs
import numpy as np
import time

from mpc import prediction
from mpc.optimization import Optimizer


class Control:
    def __init__(self, models, prediction_horizon, history_length, subsampling_rate, initial_actuators):
        self.models = models
        self.prediction_horizon = prediction_horizon
        self.history_length = history_length
        self.initial_actuators = initial_actuators
        self.interpolated_actuators = self.initial_actuators
        self.initial_optimization_values = np.tile(self.initial_actuators, prediction_horizon)
        self.subsampling_rate = subsampling_rate
        self.interpolated_tick = 0
        self.optimizer = Optimizer()
        self.predictions = list()
        prediction.initialize_models()

    def get_next_actuators(self, full_track_history, current_tick):
        # control_timer = time.perf_counter()
        print(current_tick)
        if current_tick < self.subsampling_rate:
            current_actuators = self.initial_actuators
        elif current_tick > 0 and current_tick % self.subsampling_rate == 0:

            self.interpolated_tick = current_tick % self.subsampling_rate
            history_array = prediction.prepare_tick_history(full_track_history.get_history(self.history_length, self.subsampling_rate))
            # print("control time: " + str(time.perf_counter() - control_timer) + "s")
            predictions, optimized_actuators = self.optimizer.run_optimizer(self.models,
                                                                            history_array,
                                                                            self.prediction_horizon,
                                                                            self.initial_optimization_values,
                                                                            self.history_length,
                                                                            current_tick,
                                                                            self.subsampling_rate)
            self.predictions.append(predictions)
            current_actuators = optimized_actuators[:uwabami.Tick.number_of_actuators - 1]
            self.interpolated_actuators = np.transpose(np.array([np.linspace(i, j, self.subsampling_rate + 1) for i, j in
                                                                 zip(optimized_actuators[
                                                                     :uwabami.Tick.number_of_actuators - 1],
                                                                     optimized_actuators[
                                                                     (uwabami.Tick.number_of_actuators - 1):(
                                                                        2 * (uwabami.Tick.number_of_actuators - 1))])]))
            self.initial_optimization_values = np.concatenate([optimized_actuators[3:], self.initial_actuators])
        else:
            current_actuators = self.interpolated_actuators[self.interpolated_tick]
        return current_actuators

    def write_prediction(self, index):
        matrix = self.predictions[0][0][index]
        for optimization_run in self.predictions:
            for prediction_run in optimization_run:
                matrix = np.vstack((matrix, prediction_run[index]))
        np.savetxt('prediction_' + str(index) + '.txt', matrix, delimiter=',')

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after.data[uwabami.Tick.dist_from_start_index] - myNumber < myNumber - before.data[uwabami.Tick.dist_from_start_index]:
       return after
    else:
       return before

def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid].data[uwabami.Tick.dist_from_start_index] < x: lo = mid+1
        else: hi = mid
    return lo
