import copy
import random
import itertools as iter
import math
import time
import multiprocessing as mp

from matplotlib import pyplot as plt

DATA_FOLDER = "data"


class Sudoku:
    def __init__(self, path=None):
        if path:
            f = open(path, 'r')
            lines = f.readlines()
            self.grid = map(lambda line: map(lambda elem: 0 if elem == 'x' else int(elem), line.split()), lines)
            self.final = map(lambda line: map(lambda elem: False if elem == 'x' else True, line.split()), lines)
            self.random_fill()
            self.energy = self.calculate_energy()

    def random_fill(self):
        elems_deep = [iter.repeat(num, 9) for num in xrange(1, 10)]
        elems = list(iter.chain(*elems_deep))
        for row in xrange(9):
            for column in xrange(9):
                if self.grid[row][column] != 0:
                    elems.remove(self.grid[row][column])
        random.shuffle(elems)
        for row in xrange(9):
            for column in xrange(9):
                if self.grid[row][column] == 0:
                    self.grid[row][column] = elems.pop()

    def calculate_row_energy(self, row_no):
        return 9 - len(set(map(lambda (x, y): self.grid[x][y], Sudoku.get_row_cells(row_no))))

    def calculate_column_energy(self, column_no):
        return 9 - len(set(map(lambda (x, y): self.grid[x][y], Sudoku.get_column_cells(column_no))))

    def calculate_box_energy(self, a, b):
        return 9 - len(set(map(lambda (x, y): self.grid[x][y], Sudoku.get_box_cells(a, b))))

    def calculate_energy(self):
        energy = 0
        energy += reduce(lambda x, y: x + y, [self.calculate_row_energy(r) for r in xrange(9)])
        energy += reduce(lambda x, y: x + y, [self.calculate_column_energy(c) for c in xrange(9)])
        energy += reduce(lambda x, y: x + y, [self.calculate_box_energy(a, b) for a in xrange(3) for b in xrange(3)])
        return energy

    def __deepcopy__(self, memo):
        new_sudoku = Sudoku(None)
        new_sudoku.grid = copy.deepcopy(self.grid, memo)
        new_sudoku.final = copy.deepcopy(self.final, memo)
        new_sudoku.energy = self.energy
        return new_sudoku

    @staticmethod
    def get_row_cells(row_no):
        return [(row_no, a) for a in xrange(9)]

    @staticmethod
    def get_column_cells(column_no):
        return [(a, column_no) for a in xrange(9)]

    @staticmethod
    def get_box_cells(a, b):
        return [(3 * a + i, 3 * b + j) for i in xrange(3) for j in xrange(3)]

    def not_final_in_row(self, row_no):
        return filter(lambda (x, y): not self.final[x][y], Sudoku.get_row_cells(row_no))

    def not_final_in_column(self, column_no):
        return filter(lambda (x, y): not self.final[x][y], Sudoku.get_column_cells(column_no))

    def not_final_in_box(self, a, b):
        return filter(lambda (x, y): not self.final[x][y], Sudoku.get_box_cells(a, b))

    def copy_and_permute_row(self, row_no):
        new_sudoku = copy.deepcopy(self)
        not_finals = self.not_final_in_row(row_no)
        if len(not_finals) < 2:
            return None
        else:
            [(x1, y1), (x2, y2)] = random.sample(not_finals, 2)
            new_sudoku.grid[row_no][y1], new_sudoku.grid[row_no][y2] = new_sudoku.grid[row_no][y2], \
                                                                       new_sudoku.grid[row_no][y1]
            energy_loss = self.calculate_row_energy(row_no)
            energy_loss += self.calculate_column_energy(y1)
            energy_loss += self.calculate_column_energy(y2)
            energy_loss += self.calculate_box_energy(row_no / 3, y1 / 3)
            energy_loss += self.calculate_box_energy(row_no / 3, y2 / 3)
            energy_gain = new_sudoku.calculate_row_energy(row_no)
            energy_gain += new_sudoku.calculate_column_energy(y1)
            energy_gain += new_sudoku.calculate_column_energy(y2)
            energy_gain += new_sudoku.calculate_box_energy(row_no / 3, y1 / 3)
            energy_gain += new_sudoku.calculate_box_energy(row_no / 3, y2 / 3)
            new_sudoku.energy = self.energy - energy_loss + energy_gain
            return new_sudoku

    def copy_and_permute_column(self, column_no):
        new_sudoku = copy.deepcopy(self)
        not_finals = self.not_final_in_column(column_no)
        if len(not_finals) < 2:
            return None
        else:
            [(x1, y1), (x2, y2)] = random.sample(not_finals, 2)
            new_sudoku.grid[x1][column_no], new_sudoku.grid[x2][column_no] = new_sudoku.grid[x2][column_no], \
                                                                             new_sudoku.grid[x1][column_no]
            energy_loss = self.calculate_column_energy(column_no)
            energy_loss += self.calculate_row_energy(x1)
            energy_loss += self.calculate_row_energy(x2)
            energy_loss += self.calculate_box_energy(x1 / 3, column_no / 3)
            energy_loss += self.calculate_box_energy(x2 / 3, column_no / 3)
            energy_gain = new_sudoku.calculate_column_energy(column_no)
            energy_gain += new_sudoku.calculate_row_energy(x1)
            energy_gain += new_sudoku.calculate_row_energy(x2)
            energy_gain += new_sudoku.calculate_box_energy(x1 / 3, column_no / 3)
            energy_gain += new_sudoku.calculate_box_energy(x2 / 3, column_no / 3)
            new_sudoku.energy = self.energy - energy_loss + energy_gain
            return new_sudoku

    def copy_and_permute_box(self, a, b):
        new_sudoku = copy.deepcopy(self)
        not_finals = self.not_final_in_box(a, b)
        if len(not_finals) < 2:
            return None
        else:
            [(x1, y1), (x2, y2)] = random.sample(not_finals, 2)
            new_sudoku.grid[x1][y1], new_sudoku.grid[x2][y2] = new_sudoku.grid[x2][y2], new_sudoku.grid[x1][y1]
            energy_loss = self.calculate_box_energy(a, b)
            energy_loss += self.calculate_column_energy(y1)
            energy_loss += self.calculate_column_energy(y2)
            energy_loss += self.calculate_row_energy(x1)
            energy_loss += self.calculate_row_energy(x2)
            energy_gain = new_sudoku.calculate_box_energy(a, b)
            energy_gain += new_sudoku.calculate_column_energy(y1)
            energy_gain += new_sudoku.calculate_column_energy(y2)
            energy_gain += new_sudoku.calculate_row_energy(x1)
            energy_gain += new_sudoku.calculate_row_energy(x2)
            new_sudoku.energy = self.energy - energy_loss + energy_gain
            return new_sudoku

    def copy_and_permute_random(self):
        a = random.randint(0, 2)
        if a == 0:
            result = self.copy_and_permute_row(random.randint(0, 8))
        elif a == 1:
            result = self.copy_and_permute_column(random.randint(0, 8))
        else:
            result = self.copy_and_permute_box(random.randint(0, 2), random.randint(0, 2))
        return result if result else self.copy_and_permute_random()

    def to_string(self):
        string = []
        for i in xrange(9):
            for j in xrange(9):
                string.append(chr(ord('0') + self.grid[i][j]))
            string.append('\n')
        return ''.join(string)


def simulate_annealing(sudoku, starting_temperature, final_temperature, cooling_rate, history_rate):
    if sudoku.energy == 0:
        return sudoku, []
    else:
        energy_history = []
        iteration = 0
        temperature = starting_temperature
        best_configuration = sudoku

        solved = False
        while not solved:
            try:
                if iteration % history_rate == 0:
                    energy_history.append({'iteration': iteration, 'energy': sudoku.energy})
                if temperature < final_temperature:
                    temperature = starting_temperature  # restart if solution not found
                new_sudoku = sudoku.copy_and_permute_random()
                if new_sudoku.energy < best_configuration.energy:
                    best_configuration = new_sudoku
                    sudoku = new_sudoku
                    if new_sudoku.energy == 0:
                        solved = True
                elif math.exp((sudoku.energy - new_sudoku.energy) / temperature) > random.random():
                    sudoku = new_sudoku
                temperature *= (1 - cooling_rate)
                iteration += 1
            except:
                print 'error occurred'
                temperature = starting_temperature  # restart if error

        return best_configuration, energy_history, iteration


def graph_energy_history(information, name):
    plt.figure(name)
    iterations = [x['iteration'] for x in information]
    energies = [x['energy'] for x in information]
    plt.plot(iterations, energies, '+')
    plt.savefig(name, bbox_inches='tight')


def run_task(task):
    file_input = '{}/{}/{}.txt'.format(DATA_FOLDER, task['name'], task['name'])
    file_output = '{}/{}/{}-{}-{}-{}.out'.format(DATA_FOLDER, task['name'], task['name'], task['start_temp'],
                                              task['final_temp'], task['cool_ratio'])
    result = None
    f = open(file_output, 'w')

    for i in xrange(task['try']):
        file_output_graph = '{}/{}/{}-{}-{}-{}-{}.png'.format(DATA_FOLDER, task['name'], task['name'], task['start_temp'],
                                                           task['final_temp'], task['cool_ratio'], i)
        sudoku = Sudoku(file_input)
        t0 = time.clock()
        result, energy_history, iterations = simulate_annealing(sudoku=sudoku, starting_temperature=task['start_temp'],
                                                                final_temperature=task['final_temp'],
                                                                cooling_rate=task['cool_ratio'],
                                                                history_rate=100)
        t1 = time.clock()
        graph_energy_history(energy_history, file_output_graph)
        f.write('try number: {}\nCPU time: {}\niterations: {}\n\n'.format(i, t1 - t0, iterations))
        f.flush()

    f.write(result.to_string())
    f.close()
    print 'DONE: {} \t {}'.format(task['number'], file_output)


tasks = [{'name': name, 'start_temp': start_temp, 'final_temp': final_temp, 'cool_ratio': cool_ratio, 'try': tr}
         for (name, tr) in [(1, 3), (2, 3), (3, 3), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 1), (12, 1)]
         for (start_temp, final_temp, cool_ratio) in [(40, 1e-3, 1e-4), (30, 1e-3, 1e-5), (20, 1e-3, 1e-5)]]

for i in xrange(len(tasks)):
    tasks[i]['number'] = i

pool = mp.Pool(processes=8)

pool.map(run_task, tasks)
