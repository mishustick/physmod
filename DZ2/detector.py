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



    def find_photomuls_fast(self, point):
        u_point = point[:, 2] / self.radius # по точке находим ее cos theta in [-1, 1]
        num_i = ((self.photomultiplier * u_point + self.photomultiplier - 1) // 2).astype(int) # по u восстанавливаем номер потенциального фэу
        k = int(np.ceil(self.photomultiplier * self.photocathode_r / (2*self.radius)))+1 # так как один пояс покрывается несколькими фэу, то берем Nr/2R потенциальных фэу

        close_indx_phms = np.clip(num_i[:, None] + np.arange(-k, k+1), 0, self.photomultiplier-1) # таблица с номерами потенциальных фэу, прижатая к 0 и N
        close_phms = self.pm_positions[close_indx_phms] # получили (M, 2k+1, 3) массив, каждый элемент содержит (2k+1, 3) массив с координатами потенциальных фэу

        dist_sq = (close_phms - point[:, None, :])**2
        dist_sq = np.sum(dist_sq, axis=2) # сворачиваем в сумму квадратов расстояний до фэу

        pms_min = np.argmin(dist_sq, axis=1) # находим относительный номер ближайшего фэу для каждой точки
        pms_min = np.clip(num_i - k + pms_min, 0, self.photomultiplier-1)

        min_dist_sq = np.min(dist_sq, axis=1)  # находим минимальные расстояния

        hits = min_dist_sq <= self.photocathode_r**2 # составляем маску попаданий для каждой точки
        return np.where(hits, pms_min, -1)





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

    def find_cross_point_damp(self, start, direction, lam=8):
        # Находим точку пересечения со сферой детектора с учетом затухания
        x, y, z = start
        nx, ny, nz = direction.T
        pn = x*nx + y*ny + z*nz
        t = -pn+ np.sqrt(pn**2 +self.radius**2 - x**2 - y**2 - z**2)
        cross_points = np.array(start) + t[:, None]*direction
        cross_photons = np.random.uniform(0, 1, len(cross_points)) # разыгрываем вероятность поглощения на пути
        return cross_points[cross_photons < np.exp(-t/lam) ] # возвращаем только те фотоны, которые дойдут до детектора

    def generate_photo_electrons(self, n):
        # генерация фотоэлектронов по Пуассону. Этот метод принадлежит классу
        # Detector, т.к. именно здесь реализуется моделирование детектора,
        # а -- свойство фотокатода, оно от источника не зависит
        return np.sum(np.random.poisson(self.probability_a, size=n))

