import random
import math
import Image
import copy
import multiprocessing as mp

from matplotlib import pyplot as plt


class Pixels:
    def __init__(self, size_x, size_y, pixels=None):
        self.x = size_x
        self.y = size_y
        if not pixels:
            self.pixels = [[0 for a in xrange(size_x)] for b in xrange(size_y)]
        else:
            self.pixels = pixels

    def __deepcopy__(self, memo):
        return Pixels(self.x, self.y, copy.deepcopy(self.pixels, memo))

    def fill_in_random(self, density):
        for a in xrange(int(self.x * self.y * density)):
            px = random.randint(0, self.x - 1)
            py = random.randint(0, self.y - 1)
            while self.pixels[px][py] == 1:
                px = random.randint(0, self.x - 1)
                py = random.randint(0, self.y - 1)
            self.pixels[px][py] = 1

    def calculate_energy(self, energy_function):
        energy = 0
        for i in xrange(self.x):
            for j in xrange(self.y):
                energy += energy_function(self, i, j)
        return energy


def calculate_energy_around_point(pixels, x, y, energy_function, energy_function_size):
    bounds = xrange(-(energy_function_size/2), energy_function_size/2 + 1)
    energy = 0
    for i in bounds:
        for j in bounds:
            energy += energy_function(pixels, (x + i) % pixels.x, (y + j) % pixels.y)
    return energy


def random_swap_function(old_pixels):
    px1 = random.randint(0, old_pixels.x - 1)
    py1 = random.randint(0, old_pixels.y - 1)

    px2 = random.randint(0, old_pixels.x - 1)
    py2 = random.randint(0, old_pixels.y - 1)
    while old_pixels.pixels[px1][py1] == old_pixels.pixels[px2][py2]:
        px2 = random.randint(0, old_pixels.x - 1)
        py2 = random.randint(0, old_pixels.y - 1)

    return (px1, py1), (px2, py2)


def neighbour_swap_function(old_pixels):
    swapped = False
    px1, py1, px2, py2 = 0, 0, 0, 0

    while not swapped:
        px1 = random.randint(0, old_pixels.x - 1)
        py1 = random.randint(0, old_pixels.y - 1)
        dxs = list(xrange(-1, 2))
        dys = list(xrange(-1, 2))
        random.shuffle(dxs)
        random.shuffle(dys)
        for dx in dxs:
            for dy in dys:
                px2 = (px1 + dx) % old_pixels.x
                py2 = (py1 + dy) % old_pixels.y
                if old_pixels.pixels[px1][py1] != old_pixels.pixels[px2][py2]:
                    swapped = True
    return (px1, py1), (px2, py2)


def calculate_swap(old_pixels, old_energy, energy_function, energy_function_size, swap_function):
    new_pixels = copy.deepcopy(old_pixels)
    new_energy = old_energy

    (px1, py1), (px2, py2) = swap_function(old_pixels)

    new_energy -= calculate_energy_around_point(old_pixels, px1, py1, energy_function, energy_function_size)
    new_energy -= calculate_energy_around_point(old_pixels, px2, py2, energy_function, energy_function_size)

    new_pixels.pixels[px1][py1], new_pixels.pixels[px2][py2] = old_pixels.pixels[px2][py2], old_pixels.pixels[px1][py1]

    new_energy += calculate_energy_around_point(new_pixels, px1, py1, energy_function, energy_function_size)
    new_energy += calculate_energy_around_point(new_pixels, px2, py2, energy_function, energy_function_size)

    return new_pixels, new_energy


def energy_3_3_plus(pixels, x, y):
    """
    | |x| |
    |x|o|x|
    | |x| |
    """
    pattern = [
        [-1, 01, -1],
        [01, 00, 01],
        [-1, 01, -1]
    ]
    energy = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_3_3_x(pixels, x, y):
    """
    |x| |x|
    | |o| |
    |x| |x|
    """
    pattern = [
        [01, -1, 01],
        [-1, 00, -1],
        [01, -1, 01]
    ]
    energy = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_3_3_square(pixels, x, y):
    """
    |x|x|x|
    |x|o|x|
    |x|x|x|
    """
    pattern = [
        [01, 02, 01],
        [02, 00, 02],
        [01, 02, 01]
    ]
    energy = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_5_5_grid(pixels, x, y):
    """
    |x| |x| |x|
    | |x| |x| |
    |x| |o| |x|
    | |x| |x| |
    |x| |x| |x|
    """
    pattern = [
        [02, -2, 02, -2, 02],
        [-2, 04, -4, 04, -2],
        [02, -4, 00, -4, 02],
        [-2, 04, -4, 04, -2],
        [02, -2, 02, -2, 02]
    ]
    energy = 0
    for i in [-2, -1, 0, 1, 2]:
        for j in [-2, -1, 0, 1, 2]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_5_5_ball(pixels, x, y):
    """
    | | |x| | |
    | |x|x|x| |
    |x|x|o|x|x|
    | |x|x|x| |
    | | |x| | |
    """
    pattern = [
        [00, 00, 01, 00, 00],
        [00, 01, 02, 01, 00],
        [01, 02, 00, 02, 01],
        [00, 01, 02, 01, 00],
        [00, 00, 01, 00, 00]
    ]
    energy = 0
    for i in [-2, -1, 0, 1, 2]:
        for j in [-2, -1, 0, 1, 2]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_7_7_x(pixels, x, y):
    """
    |x| | | | | |x|
    | |x| | | |x| |
    | | |x| |x| | |
    | | | |x| | | |
    | | |x| |x| | |
    | |x| | | |x| |
    |x| | | | | |x|
    """
    pattern = [
        [04, 02, -1, -4, -1, 02, 04],
        [02, 04, 02, -1, 02, 04, 02],
        [-1, 02, 04, 02, 04, 02, -1],
        [-4, -1, 02, 00, 02, -1, -4],
        [-1, 02, 04, 02, 04, 02, -1],
        [02, 04, 02, -1, 02, 04, 02],
        [04, 02, -1, -4, -1, 02, 04]
    ]
    energy = 0
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_7_7_verticals_thin(pixels, x, y):
    """
    | |x| |x| |x| |
    | |x| |x| |x| |
    | |x| |x| |x| |
    | |x| |o| |x| |
    | |x| |x| |x| |
    | |x| |x| |x| |
    | |x| |x| |x| |
    """
    pattern = [
        [-2, 02, -2, 02, -2, 02, -2],
        [-2, 02, -2, 02, -2, 02, -2],
        [-2, 02, -2, 02, -2, 02, -2],
        [-2, 02, -2, 00, -2, 02, -2],
        [-2, 02, -2, 02, -2, 02, -2],
        [-2, 02, -2, 02, -2, 02, -2],
        [-2, 02, -2, 02, -2, 02, -2]
    ]
    energy = 0
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_7_7_verticals_thick(pixels, x, y):
    """
    | | |x|x|x| | |
    | | |x|x|x| | |
    | | |x|x|x| | |
    | | |x|o|x| | |
    | | |x|x|x| | |
    | | |x|x|x| | |
    | | |x|x|x| | |
    """
    pattern = [
        [-4, -2, 02, 04, 02, -2, -4],
        [-4, -2, 02, 04, 02, -2, -4],
        [-4, -2, 02, 04, 02, -2, -4],
        [-4, -2, 02, 04, 02, -2, -4],
        [-4, -2, 02, 04, 02, -2, -4],
        [-4, -2, 02, 04, 02, -2, -4],
        [-4, -2, 02, 04, 02, -2, -4]
    ]
    energy = 0
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_7_7_circle(pixels, x, y):
    """
    | | | |x| | | |
    | |x|x|x|x|x| |
    | |x|x|x|x|x| |
    |x|x|x|o|x|x|x|
    | |x|x|x|x|x| |
    | |x|x|x|x|x| |
    | | | |x| | | |
    """
    pattern = [
        [00, 00, 00, 01, 00, 00, 00],
        [00, 01, 01, 02, 01, 01, 00],
        [00, 01, 02, 04, 02, 01, 00],
        [01, 02, 04, 00, 04, 02, 01],
        [00, 01, 02, 04, 02, 01, 00],
        [00, 01, 01, 02, 01, 01, 00],
        [00, 00, 00, 01, 00, 00, 00]
    ]
    energy = 0
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_7_7_v(pixels, x, y):
    """
    |x| | | | | |x|
    |x|x| | | |x|x|
    |x|x|x| |x|x|x|
    | |x|x|o|x|x| |
    | | |x|x|x| | |
    | | | |x| | | |
    | | | | | | | |
    """
    pattern = [
        [01, 00, -1, -2, -1, 00, 01],
        [02, 01, 00, -1, 00, 01, 02],
        [01, 02, 01, 00, 01, 02, 01],
        [00, 01, 02, 00, 02, 01, 00],
        [00, 00, 01, 02, 01, 00, 00],
        [-1, 00, 00, 01, 00, 00, -1],
        [-2, -1, 00, 00, 00, -1, -2]
    ]
    energy = 0
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_7_7_slash(pixels, x, y):
    """
    | | | | | |x|x|
    | | | | |x|x|x|
    | | | |x|x|x| |
    | | |x|o|x| | |
    | |x|x|x| | | |
    |x|x|x| | | | |
    |x|x| | | | | |
    """
    pattern = [
        [-2,  -1,  -2,  00,  00,  01,  02],
        [-1,  -1,  00,  00,  01,  02,  01],
        [-1,  00,  00,  01,  02,  01,  00],
        [00,  00,  01,  00,  01,  00,  00],
        [00,  01,  02,  01,  00,  00,  -1],
        [01,  02,  01,  00,  00,  -1,  -1],
        [02,  01,  00,  00,  -1,  -1,  -2]
    ]
    energy = 0
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy -= pattern[i][j]
            else:
                energy += pattern[i][j]

    return energy


def energy_9_9_labirynth(pixels, x, y):
    """
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | |x|x|x| | | |
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | |x|x|x| | | |
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | |x|x|x| | | |
    | | | |x|x|x| | | |     |x|x|x|x|x|x|x|x|x|     | | | |x|x|x|x|x|x|
    | | | |x|x|x| | | |     |x|x|x|x|x|x|x|x|x|     | | | |x|x|x|x|x|x|
    | | | |x|x|x| | | |     |x|x|x|x|x|x|x|x|x|     | | | |x|x|x|x|x|x|
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | | | | | | | |
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | | | | | | | |
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | | | | | | | |

    | | | |x|x|x| | | |     | | | | | | | | | |     | | | | | | | | | |
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | | | | | | | |
    | | | |x|x|x| | | |     | | | | | | | | | |     | | | | | | | | | |
    |x|x|x|x|x|x| | | |     |x|x|x|x|x|x| | | |     | | | |x|x|x|x|x|x|
    |x|x|x|x|x|x| | | |     |x|x|x|x|x|x| | | |     | | | |x|x|x|x|x|x|
    |x|x|x|x|x|x| | | |     |x|x|x|x|x|x| | | |     | | | |x|x|x|x|x|x|
    | | | | | | | | | |     | | | |x|x|x| | | |     | | | |x|x|x| | | |
    | | | | | | | | | |     | | | |x|x|x| | | |     | | | |x|x|x| | | |
    | | | | | | | | | |     | | | |x|x|x| | | |     | | | |x|x|x| | | |

    """
    pattern1 = [
        [-2, -2, -2, 02, 04, 02, -2, -2, -2],
        [-2, -2, -2, 02, 04, 02, -2, -2, -2],
        [-2, -2, -2, 02, 04, 02, -2, -2, -2],
        [00, 00, 00, 02, 04, 02, 00, 00, 00],
        [00, 00, 00, 02, 00, 02, 00, 00, 00],
        [00, 00, 00, 02, 04, 02, 00, 00, 00],
        [-2, -2, -2, 02, 04, 02, -2, -2, -2],
        [-2, -2, -2, 02, 04, 02, -2, -2, -2],
        [-2, -2, -2, 02, 04, 02, -2, -2, -2]
    ]

    pattern2 = [
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [02, 02, 02, 02, 02, 02, 02, 02, 02],
        [04, 04, 04, 04, 04, 04, 04, 04, 04],
        [02, 02, 02, 02, 02, 02, 02, 02, 02],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2]
    ]

    pattern3 = [
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [00, 00, 00, 02, 05, 02, 02, 02, 02],
        [00, 00, 00, 02, 05, 05, 05, 05, 05],
        [00, 00, 00, 02, 02, 02, 02, 02, 02],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2]
    ]

    pattern4 = [
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [02, 02, 02, 02, 05, 02, 00, 00, 00],
        [05, 05, 05, 05, 05, 02, 00, 00, 00],
        [02, 02, 02, 02, 02, 02, 00, 00, 00],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2]
    ]

    pattern5 = [
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [02, 02, 02, 02, 02, 02, 00, 00, 00],
        [05, 05, 05, 05, 05, 02, 00, 00, 00],
        [02, 02, 02, 02, 05, 02, 00, 00, 00],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2]
    ]

    pattern6 = [
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [-2, -2, -2, 00, 00, 00, -2, -2, -2],
        [00, 00, 00, 02, 02, 02, 02, 02, 02],
        [00, 00, 00, 02, 05, 05, 05, 05, 05],
        [00, 00, 00, 02, 05, 02, 02, 02, 02],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2],
        [-2, -2, -2, 02, 05, 02, -2, -2, -2]
    ]

    energy1 = 0
    energy2 = 0
    energy3 = 0
    energy4 = 0
    energy5 = 0
    energy6 = 0
    for i in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        for j in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            if pixels.pixels[x][y] == pixels.pixels[(x + i) % pixels.x][(y + j) % pixels.y]:
                energy1 -= pattern1[i][j]
                energy2 -= pattern2[i][j]
                energy3 -= pattern3[i][j]
                energy4 -= pattern4[i][j]
                energy5 -= pattern5[i][j]
                energy6 -= pattern6[i][j]
            else:
                energy1 += pattern1[i][j]
                energy2 += pattern2[i][j]
                energy3 += pattern3[i][j]
                energy4 += pattern4[i][j]
                energy5 += pattern5[i][j]
                energy6 += pattern6[i][j]

    return min([energy1, energy2, energy3, energy4, energy5, energy6])


def simulate_annealing(pixels, starting_temperature, final_temperature, cooling_rate, energy_function,
                       energy_function_size, swap_function):
    temperature = starting_temperature
    energy = pixels.calculate_energy(energy_function)
    best_configuration = {'energy': energy, 'pixels': pixels}

    while final_temperature < temperature:
        new_pixels, new_energy = calculate_swap(pixels, energy, energy_function, energy_function_size, swap_function)

        if new_energy < best_configuration['energy']:
            best_configuration = {'energy': new_energy, 'pixels': new_pixels}
            pixels = new_pixels
            energy = new_energy
        elif math.exp((energy - new_energy) / temperature) > random.random():
            pixels = new_pixels
            energy = new_energy

        temperature *= (1 - cooling_rate)

    return best_configuration['pixels']


def graph_energy_history(information):
    iterations = [x['iteration'] for x in information]
    energies = [x['energy'] for x in information]
    plt.plot(iterations, energies, '+')
    plt.show()


def save_pixels(pixels, name):
    img = Image.new('1', (pixels.x, pixels.y), color=1)
    for i in xrange(pixels.x):
        for j in xrange(pixels.y):
            if pixels.pixels[i][j] == 1:
                img.putpixel((i, j), 0)
    img.save(name + '.bmp', 'BMP')


def run_task(task):
    pixels = Pixels(task['size'], task['size'])
    pixels.fill_in_random(task['density'])
    save_pixels(pixels, task['name'] + '-start')
    best_configuration = simulate_annealing(pixels=pixels,
                                            starting_temperature=task['start_temp'],
                                            final_temperature=task['final_temp'],
                                            cooling_rate=task['cool_rate'],
                                            energy_function=task['energy_func'],
                                            energy_function_size=task['energy_size'],
                                            swap_function=random_swap_function)
    save_pixels(best_configuration, task['name'] + '-end')

tasks = [{'name':  "{0}-{1}-{2}".format(energy_func.__name__, str(size), str(density)[2]),
          'density': density,
          'size': size,
          'start_temp': start_temp,
          'final_temp': final_temp,
          'cool_rate': cool_rate,
          'energy_func': energy_func,
          'energy_size': energy_size}
         for density in [0.1, 0.3, 0.4, 0.6]
         for size in [50, 250]
         for start_temp in [100]
         for final_temp in [1]
         for cool_rate in [1e-4]
         for (energy_func, energy_size) in
         [(energy_3_3_plus, 3), (energy_3_3_x, 3), (energy_3_3_square, 3), (energy_5_5_grid, 5),
          (energy_5_5_ball, 5), (energy_7_7_circle, 7), (energy_7_7_slash, 7), (energy_7_7_v, 7),
          (energy_7_7_verticals_thick, 7), (energy_7_7_verticals_thin, 7), (energy_9_9_labirynth, 9)]]


pool = mp.Pool(processes=8)
pool.map(run_task, tasks)
