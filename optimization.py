# python libs
from scipy import optimize
import numpy as np

from prediction import prediction

DISTANCE_THRESHOLD_SOFT = 0.6
DISTANCE_THRESHOLD_HARD = 0.9
BAD_FITNESS = 1

STEER_BOUNDS = [-0.5, 0.5]
ACCEL_BOUNDS = [0.3, 0.5]
BRAKE_BOUNDS = [0.0, 0.4]
BOUNDS_LIST = STEER_BOUNDS, ACCEL_BOUNDS, BRAKE_BOUNDS


class Optimizer:
    def __init__(self):
        self.predictions = list()

    def run_optimizer(self, models, history_array, prediction_horizon, initial_guess, history_length, initial_tick, subsampling_rate, number_of_controls, number_of_variables):
        self.predictions = list()
        x, f, d = optimize.fmin_l_bfgs_b(func=self.evaluate_fitness,
                                         x0=initial_guess,
                                         bounds=BOUNDS_LIST * prediction_horizon,
                                         approx_grad=True,
                                         args=(history_array, models, prediction_horizon, history_length, initial_tick, number_of_controls, number_of_variables),
                                         maxfun=280,
                                         iprint=-1,
                                         factr=1e20)
        return self.predictions, x

    def evaluate_fitness(self, actuators, history_array, models, prediction_horizon, history_length, initial_tick):
        predictions = prediction(models, actuators, history_array, prediction_horizon, history_length, initial_tick)

        self.predictions.append(predictions)

        # Minimize the distance from the center - this fitness function must be replaced according to the context
        # fitness = np.amax(np.abs(np.array(predictions[uwabami.Tick.center_index])))
        # return fitness
        pass # placeholder
