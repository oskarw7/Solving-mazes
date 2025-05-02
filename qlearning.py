from typing import List, Tuple, Optional, Any
import time
import random
import numpy as np
import pickle

class Model:
    def __init__(self, matrix, entry, goal):
        self.matrix = matrix
        self.entry = entry
        self.goal = goal

        self.width = len(matrix[0])
        self.height = len(matrix)

        self.qtable = [[[0.0 for _ in range(4)] for _ in range(self.width)] for _ in range(self.height)]
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.learning_rate = 0.1
        self.discount_factor = 0.99

        self.epsilon_decay = 0.998

    def learn(self, episodes : int) -> tuple[float, int | Any]:

        startTime = time.time()

        nodesVisited = 0
        epsilon = 1
        max_steps = self.width * self.height * 2
        reward = self.width * 100


        for ep in range(episodes):
            position = self.entry
            steps = 0
            epsilon = max(0.01, epsilon * self.epsilon_decay)

            while position != self.goal:
                x, y = position
                nodesVisited += 1
                steps += 1

                if steps > max_steps:
                    print(f"Episode {ep + 1}: Max steps reached")
                    break

                if random.random() < epsilon:
                    move_idx = random.randint(0, 3)
                else:
                    qvals = self.qtable[y][x]
                    max_q = max(qvals)
                    best_moves = [i for i, val in enumerate(qvals) if val == max_q]
                    move_idx = random.choice(best_moves)

                d_x, d_y = self.directions[move_idx]
                next_x = x + d_x
                next_y = y + d_y

                old_dist = abs(self.goal[0] - x) + abs(self.goal[1] - y)
                new_dist = abs(self.goal[0] - next_x) + abs(self.goal[1] - next_y)

                if not (0 <= next_x < self.width and 0 <= next_y < self.height):
                    r = -10
                    next_x, next_y = x, y
                elif self.matrix[next_y][next_x] == 1:
                    r = -10
                    next_x, next_y = x, y
                elif (next_x, next_y) == self.goal:
                    r = reward
                else:
                    r = -1 + (old_dist - new_dist)

                next_max = max(self.qtable[next_y][next_x])
                prev_value = self.qtable[y][x][move_idx]
                self.qtable[y][x][move_idx] += self.learning_rate * (r + self.discount_factor * next_max - prev_value)

                position = (next_x, next_y)

            if not (ep + 1) % (episodes // 10):
                print(f"Episode {ep + 1}/{episodes} completed")

        return time.time() - startTime, nodesVisited


    def run(self):
        startTime = time.time()
        position = self.entry
        path = [self.entry]
        steps = 0
        max_steps = self.width * self.height * 2

        while position != self.goal and steps < max_steps:
            x, y = position
            steps += 1

            move_idx = int(np.argmax(self.qtable[y][x]))
            d_x, d_y = self.directions[move_idx]
            next_x, next_y = x + d_x, y + d_y

            if not (0 <= next_x < self.width and 0 <= next_y < self.height):
                break
            if self.matrix[next_y][next_x] != 0:
                break

            position = (next_x, next_y)
            path.append(position)

        if position != self.goal:
            return None, time.time() - startTime, steps, self.qtable

        return path, time.time() - startTime, steps, self.qtable

    def serialize(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.qtable, f)

    def unserialize(self, filename):
        with open(filename, "rb") as f:
            self.qtable = pickle.load(f)