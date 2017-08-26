# python libs
import numpy as np

import prediction
from optimization import Optimizer


class Control:
    def __init__(self, models, prediction_horizon, history_length, subsampling_rate, initial_controls, number_of_controls, number_of_variables):
        self.models = models
        self.prediction_horizon = prediction_horizon
        self.history_length = history_length
        self.initial_controls = initial_controls
        self.interpolated_controls = self.initial_controls
        self.initial_optimization_values = np.tile(self.initial_controls, prediction_horizon)
        self.subsampling_rate = subsampling_rate
        self.number_of_controls = number_of_controls
        self.number_of_variables = number_of_variables
        self.interpolated_tick = 0
        self.optimizer = Optimizer()
        self.predictions = list()
        prediction.initialize_models()

    def get_next_actuators(self, full_track_history, current_tick):
        print(current_tick)
        if current_tick < self.subsampling_rate:
            current_actuators = self.initial_controls
        elif current_tick > 0 and current_tick % self.subsampling_rate == 0:

            self.interpolated_tick = current_tick % self.subsampling_rate
            history_array = prediction.prepare_tick_history(full_track_history.get_history(self.history_length, self.subsampling_rate))
            predictions, optimized_actuators = self.optimizer.run_optimizer(self.models,
                                                                            history_array,
                                                                            self.prediction_horizon,
                                                                            self.initial_optimization_values,
                                                                            self.history_length,
                                                                            current_tick,
                                                                            self.subsampling_rate,
                                                                            self.number_of_controls,
                                                                            self.number_of_variables)
            self.predictions.append(predictions)
            current_actuators = optimized_actuators[:self.number_of_controls - 1]
            self.interpolated_controls = np.transpose(np.array([np.linspace(i, j, self.subsampling_rate + 1) for i, j in
                                                                 zip(optimized_actuators[
                                                                     :self.number_of_controls - 1],
                                                                     optimized_actuators[
                                                                     (self.number_of_controls - 1):(
                                                                        2 * (self.number_of_controls - 1))])]))
            self.initial_optimization_values = np.concatenate([optimized_actuators[3:], self.initial_controls])
        else:
            current_actuators = self.interpolated_controls[self.interpolated_tick]
        return current_actuators

    def write_prediction(self, index):
        matrix = self.predictions[0][0][index]
        for optimization_run in self.predictions:
            for prediction_run in optimization_run:
                matrix = np.vstack((matrix, prediction_run[index]))
        np.savetxt('prediction_' + str(index) + '.txt', matrix, delimiter=',')
