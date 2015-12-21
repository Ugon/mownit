import random
import math
import argparse

from matplotlib import pyplot as plt


def gauss_in_section(a, b, grouping_param):
    result = random.gauss(float(a + b) / 2, float(b - a) / grouping_param)
    if a < result < b:
        return result
    else:
        return gauss_in_section(a, b, grouping_param)


def random_uniform(number, dx, dy):
    points = []
    for i in xrange(number):
        points.append((random.uniform(-dx, dx), random.uniform(-dy, dy)))
    random.shuffle(points)
    return points


def random_normal_separated_nine_groups(number_in_grp, dx, dy):
    grp_size_x = float(2 * dx) / 5
    grp_size_y = float(2 * dy) / 5
    points = []
    for pt in xrange(number_in_grp):
        for i in xrange(3):
            for j in xrange(3):
                px = gauss_in_section(i * 2 * grp_size_x, i * 2 * grp_size_x + grp_size_x, 2.5)
                py = gauss_in_section(j * 2 * grp_size_y, j * 2 * grp_size_y + grp_size_y, 2.5)
                point = (px - dx, py - dy)
                points.append(point)
    random.shuffle(points)
    return points


def random_normal_four_groups(number_in_grp, dx, dy):
    points = []
    for pt in xrange(number_in_grp):
        for i in [-1, 0]:
            for j in [-1, 0]:
                px = gauss_in_section(i * dx, i * dx + dx, 6)
                py = gauss_in_section(j * dy, j * dy + dy, 6)
                point = (px, py)
                points.append(point)
    random.shuffle(points)
    return points


def distance((a, b), (c, d)):
    return (a - c) ** 2 + (b - d) ** 2


def configuration_energy(cities):
    return sum([distance(a, b) for (a, b) in zip(cities, cities[1:])])


def consecutive_swap(cities):
    new_cities = list(cities)
    l = len(cities)
    i = random.randint(0, l - 1)
    j = (i + 1) % l
    new_cities[i], new_cities[j] = new_cities[j], new_cities[i]
    return new_cities


def arbitrary_swap(cities):
    new_cities = list(cities)
    l = len(cities)
    j = i = random.randint(0, l - 1)
    while j == i:
        j = random.randint(0, l - 1)

    new_cities[i], new_cities[j] = new_cities[j], new_cities[i]
    return new_cities


def simulate_annealing(cities, max_iterations, starting_temperature, cooling_rate, swap_type, history_rate):
    energy_history = []
    best_configuration_history = []

    iteration = 0
    energy = configuration_energy(cities)
    temperature = starting_temperature

    best_configuration_history.append({'energy': energy, 'iteration': iteration, 'cities': cities})

    for i in xrange(max_iterations):
        if iteration % history_rate == 0:
            energy_history.append({'iteration': iteration, 'energy': energy})

        if swap_type == 'arbitrary':
            new_cities = arbitrary_swap(cities)
        else:
            new_cities = consecutive_swap(cities)

        new_energy = configuration_energy(new_cities)
        if new_energy < best_configuration_history[-1]['energy']:
            best_configuration_history.append({'energy': new_energy, 'iteration': iteration, 'cities': new_cities})
            cities = new_cities
            energy = new_energy
        elif math.exp((energy - new_energy) / temperature) > random.random():
            cities = new_cities
            energy = new_energy

        temperature *= (1 - cooling_rate)
        iteration += 1

    return best_configuration_history[-1]['cities'], energy_history, best_configuration_history


def graph_energy_history(information):
    iterations = [x['iteration'] for x in information]
    energies = [x['energy'] for x in information]
    plt.plot(iterations, energies, '+')
    plt.show()


def plot_result(cities, dx, dy):
    fig = plt.figure()
    canvas = fig.add_subplot(1, 1, 1)

    canvas.clear()
    canvas.set_xlim(-dx, dx)
    canvas.set_ylim(-dy, dy)

    for ((x1, y1), (x2, y2)) in zip(cities, cities[1:]):
        plt.plot([x1, x2], [y1, y2], color='b', linestyle='-', linewidth=0.4, marker='o')

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-generation', type=str, required=True, choices=['1', '4', '9'])
parser.add_argument('-num_of_cities_in_grp', type=int, required=True, dest='num_of_cities_in_grp')
parser.add_argument('-dx', type=float, required=True, dest='dx')
parser.add_argument('-dy', type=float, required=True, dest='dy')
parser.add_argument('-start_temp', type=float, required=True, dest='start_temp')
parser.add_argument('-cooling_rate', type=float, required=True, dest='cooling_rate')
parser.add_argument('-max_iterations', type=int, required=True, dest='max_iterations')
parser.add_argument('-swap_type', type=str, required=True, default='consecutive', dest='swap_type',
                    choices=['arbitrary', 'consecutive'])

args = vars(parser.parse_args())
generation = args['generation']
num_of_cities_in_grp = args['num_of_cities_in_grp']
dx = args['dx']
dy = args['dy']
start_temp = args['start_temp']
cooling_rate = args['cooling_rate']
max_iterations = args['max_iterations']
swap_type = args['swap_type']

if generation == '1':
    cities = random_uniform(num_of_cities_in_grp, dx, dy)
elif generation == '4':
    cities = random_normal_four_groups(num_of_cities_in_grp, dx, dy)
elif generation == '9':
    cities = random_normal_separated_nine_groups(num_of_cities_in_grp, dx, dy)
else:
    cities = []
    print "ERROR"
    exit()

cities, energy_history, best_configuration_history = \
    simulate_annealing(cities=cities, max_iterations=max_iterations, starting_temperature=start_temp,
                       cooling_rate=cooling_rate, swap_type=swap_type, history_rate=1)

plot_result(cities, 10, 10)

graph_energy_history(energy_history)
