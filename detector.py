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
        phi = i * np.pi * (3 - np.sqrt(5)) # размещение по сфере Фибоначчи

        self.pm_positions = np.array([self.radius * np.sqrt(1 - u**2) * np.cos(phi), self.radius * np.sqrt(1- u**2) * np.sin(phi), self.radius * u])
        self.pm_positions = self.pm_positions.transpose()


    def find_photomultiplier(self, point):
        # вычисление квадрата расстояний от точки до фэу
        distances_squared = (self.pm_positions[:, None, :] - point[None, :, :])**2 # векторная разность точек и фэу
        distances_squared = np.sum(distances_squared, axis=2) # суммирование квадратов компонент разности
        pm_min = np.argmin(distances_squared, axis=0) # поиск для каждой точки ближайшего фэу
        min_dist_sq = np.min(distances_squared, axis=0) # поиск минимальных расстояний

        hit = min_dist_sq <= self.photocathode_r**2 # проверка попадания
        return np.where(hit, pm_min, -1)

    def find_cross_point(self, start, direction):
        # Находим точку пересечения со сферой детектора
        x, y, z = start
        nx, ny, nz = direction.T
        pn = x*nx + y*ny + z*nz
        t = -pn+ np.sqrt(pn**2 +self.radius**2 - x**2 - y**2 - z**2)
        cross_points = np.array(start) + t[:, None]*direction
        return cross_points

    def generate_photo_electrons(self, n):
        # генерация фотоэлектронов по Пуассону. Этот метод принадлежит классу
        # Detector, т.к. именно здесь реализуется моделирование детектора,
        # а -- свойство фотокатода, оно от источника не зависит
        return np.sum(np.random.poisson(self.probability_a, size=n))

