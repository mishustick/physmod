import numpy as np


class Source:
    def __init__(self, Y):
        self.light_yield = Y

    def generate_directions(self, energy):
        N = round(self.light_yield * energy)
        # изотропное случайное распределение
        phi = np.random.uniform(0, 2*np.pi, N)
        u = np.random.uniform(-1, 1, N) # cos theta

        normal_vectors = np.array([np.sqrt(1 - u**2)*np.cos(phi), np.sqrt(1 - u**2)*np.sin(phi), u]).transpose()

        return normal_vectors
