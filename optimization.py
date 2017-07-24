# python libs
from scipy import optimize
import numpy as np
import time

# uwabami libs
from mpc.prediction import prediction
from uwabami import uwabami

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

    def run_optimizer(self, models, history_array, prediction_horizon, initial_guess, history_length, initial_tick, subsampling_rate):
        # optimization_timer = time.perf_counter()
        self.predictions = list()
        x, f, d = optimize.fmin_l_bfgs_b(func=self.evaluate_fitness,
                                         x0=initial_guess,
                                         bounds=BOUNDS_LIST * prediction_horizon,
                                         approx_grad=True,
                                         args=(history_array, models, prediction_horizon, history_length, initial_tick),
                                         maxfun=280,
                                         iprint=-1,
                                         factr=1e20)
        # x = optimize.fmin_slsqp(func=evaluate_fitness,
        #                         x0=initial_guess,
        #                         bounds=BOUNDS_LIST*prediction_horizon,
        #                         iter=100,
        #                         args=(history, models, prediction_horizon, history_length, initial_tick))

        # print("optimization time: " + str(time.perf_counter() - optimization_timer) + "s")
        return self.predictions, x

    def evaluate_fitness(self, actuators, history_array, models, prediction_horizon, history_length, initial_tick):
        # prediction_timer = time.perf_counter()
        predictions = prediction(models, actuators, history_array, prediction_horizon, history_length, initial_tick)
        # print("prediction time: " + str(time.perf_counter() - prediction_timer) + "s")

        # print(predictions[uwabami.Tick.center_index])
        # print(predictions[uwabami.Tick.dist_from_start_index])
        # if the vehicle left the track, return really high value
        # if (np.abs(predictions[uwabami.Tick.center_index]) > DISTANCE_THRESHOLD_HARD).sum() > 0:
        #     fitness = BAD_FITNESS
        # else return a combination of distance to center and top speed, in order to account for both
        # elif (np.abs(predictions[uwabami.Tick.speed_x_index]) < 0).sum() > 0:
        #     fitness = BAD_FITNESS
        # else:
        #     violations = np.asarray(
        #         [pow((abs(x) - DISTANCE_THRESHOLD_SOFT) * 10, 2) for x in predictions[uwabami.Tick.center_index]])
        #     fitness = -1 / (np.mean(np.subtract(predictions[uwabami.Tick.speed_x_index], violations)))
            # fitness = abs(1/np.mean(speed))
            # fitness = -np.amin(predictions[uwabami.Tick.speed_x_index])
            # fitness = -predictions[uwabami.Tick.dist_from_start_index][-1]
        # print("## FITNESS ##: " + str(fitness))
        # print("####################")
        # fitness = np.max(np.abs(dist)) - np.min(np.abs(speed))/300
        # print(fitness)
        # min_speed = np.min(predictions[uwabami.Tick.speed_x_index])

        self.predictions.append(predictions)

        # last_dist = (predictions[uwabami.Tick.dist_from_start_index][-1])
        # predictions[uwabami.Tick.center_index][np.abs(predictions[uwabami.Tick.center_index]) < DISTANCE_THRESHOLD_HARD] = 0
        # penalty = np.mean(predictions[uwabami.Tick.center_index] * predictions[uwabami.Tick.dist_from_start_index])
        # fitness = penalty - last_dist
        # # print(fitness)
        # return fitness

        # Minimize the distance from the center
        fitness = np.amax(np.abs(np.array(predictions[uwabami.Tick.center_index])))
        print(predictions[uwabami.Tick.center_index])
        print(fitness)
        # last_dist = (predictions[uwabami.Tick.dist_from_start_index][-1])
        # # mean_speed = np.mean(predictions[uwabami.Tick.speed_x_index])
        # max_center = np.max(np.abs(predictions[uwabami.Tick.center_index]))
        # # fitness = -abs(mean_speed) * (1 - pow(max_center, 2)) * (1.5 + 0.5 * min_speed/abs(min_speed))
        # fitness = abs(last_dist) * (pow(max_center, 2) - 1)
        return fitness
