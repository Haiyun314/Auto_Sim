import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time


def timeit(func):
    def inner(*args):
        start = time.perf_counter()
        result = func(*args)
        print(f'time cost {time.perf_counter() - start} seconds')
        return result
    return inner


class Alg:
    @staticmethod
    @timeit
    def distance(sensor_front, sensor_end, map, shape):
        """
        use binary search to find the distance to object
        :param sensor_front: sensor's position
        :param sensor_end: the end position can be detected
        :param map: running area
        :return: the distance from sensor to the object
        """
        if np.linalg.norm(sensor_end - sensor_front) > 2: # the error is smaller than 2
            if any(sensor_end < 0) or any(sensor_end > shape):
                sensor_end = np.round((sensor_front + sensor_end) / 2).astype(int)
            else:
                if map[tuple(sensor_end)]:
                    sensor_end = np.round((sensor_front + sensor_end) / 2).astype(int)
                else:
                    sensor_front = np.round((sensor_front + sensor_end) / 2).astype(int)
            return Alg.distance(sensor_front, sensor_end, map, shape)
        else:
            return sensor_end


class Sensor(Alg):
    def __init__(self, car_size, number_of_sensors, map):
        self.car_size = car_size
        self.number_of_sensors = number_of_sensors
        self.map = map

    def object_distance(self, sensor_front, sensor_end):
        """
        :param sensor_front: position of sensor sender, [x, y]
        :param sensor_end: position of detection end point from different direction, [x, y]
        :return:
        """
        assert isinstance(sensor_front, np.ndarray), 'data type should be a np.array()'
        assert isinstance(sensor_end, np.ndarray), 'data type should be a np.array()'
        shape = self.map.shape
        distance = Alg.distance(sensor_front, sensor_end, self.map, shape)
        return distance

class Car:
    """
    radar_numbers : numbers of radars on the car
    time_interval : 1 over FPS
    """
    radar_numbers = 10
    time_interval = 0.01

    def __init__(self, car_size, number_of_sensors, _map, init_velocity, position, map):
        super().__init__(car_size, number_of_sensors, map)
        self.car_size = car_size
        self.init_velocity = init_velocity
        self.position = position

    @staticmethod
    def rotation_matrix(steer, theta):
        rotation_matrix = np.array([[np.cos(steer * theta), -np.sin(steer * theta)],
                                    [np.sin(steer * theta, np.cos(steer * theta))]])
        return rotation_matrix

    def control(self, velocity, accelaration, steer, direction):
        assert -1 < accelaration < 2, 'acceleration is bounded from (-1, 2)'
        assert -1 < steer < 1, '-1 means totally left side 45 degree, 1 means totally right side 45 degree'
        velocity = velocity + accelaration * (self.time_interval) * self.rotation_matrix(steer, 45) * direction
        return velocity

    def sensor(self, car_center, detection_distance):
        # radars = {}
        # for radar in range(self.radar_numbers):
        #     radars += {f'radar{radar}': (radar_position, detection_distance * self.control() * self.control())}
        pass


class Driving_NN:
    def __init__(self):
        pass

    def model(self, current_position, velocity, sensor_information):
        """
        a DNN model which take inputs: current position, velocity and sensor information
        :return: model : next frame acceleration, steer
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer())
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(2))
        return model


def run(model, position, velocity, sensor_information):
    """
    predict the acceleration and steer for the next frame
    :param position: current position
    :param velocity: current velocity
    :param radar_information: current sensor_information
    :return:
    """
    acceleration, steer = model.predict(position, velocity, sensor_information)
    return acceleration, steer


def main():
    image = plt.imread(PATH + '/map/map.png')[..., 0]
    image = image.transpose()
    print(image.shape)
    print(Alg.distance(np.array([1, 4]), np.array([34, -10]), image, np.array([800, 600])))

    plt.imshow(image)
    plt.show()
    print(image)


if __name__ == '__main__':
    PATH = os.getcwd()
    main()
