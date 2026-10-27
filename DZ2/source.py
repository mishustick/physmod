import numpy as np


class Source:
    def __init__(self, Y, position):
        self.light_yield = Y
        self.position = position # из какой точки летят фотоны

    # def generate_directions(self, energy):
    #     N = np.random.poisson(energy*self.light_yield) # теперь случайное число фотонов



    def generate_photons(self, energy, number_photons=1): # задаем по сколько фотонов генерировать
        N = np.random.poisson(energy * self.light_yield)  # теперь случайное число фотонов
        # изотропное случайное распределение
        n_left = N
        while n_left > 0:
            number_gen = min(number_photons, n_left)
            phi = np.random.uniform(0, 2 * np.pi, number_gen)
            u = np.random.uniform(-1, 1, number_gen)  # cos theta
            n_left -= number_gen
            normal_vectors = np.array([np.sqrt(1 - u ** 2) * np.cos(phi), np.sqrt(1 - u ** 2) * np.sin(phi), u]).transpose()
            yield normal_vectors