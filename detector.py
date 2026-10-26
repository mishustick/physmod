import numpy as np

class Detector:
    def __init__(self, radius, photomultiplier, photocathode_r, probability_a):
        self.radius = radius
        self.photomultiplier = photomultiplier
        self.photocathode_r = photocathode_r
        self.probability_a = probability_a

    def place_photomultipliers(self):
        # Расставляем равномерно фэу
        N = self.photomultiplier
        i = np.arange(N)
        u = -1 + (2 * i + 1) / N
        phi = i * np.pi * (3 - np.sqrt(5)) # размещение по сфере Фибоначии

        self.pm_positions = np.array([self.radius * np.sqrt(1 - u**2) * np.cos(phi), self.radius * np.sqrt(1- u**2) * np.sin(phi), self.radius * u])
        self.pm_positions = self.pm_positions.transpose()


    def find_photomultiplier(self, point):
        # вычисление квадрата расстояний от точки до фэу
        distances_squared = (self.pm_positions - point)**2
        distances_squared = np.sum(distances_squared, axis=1)
        pm_min = np.argmin(distances_squared)
        if distances_squared[pm_min] <= self.photocathode_r**2:
            return pm_min
        else:
            return None

