from typing import List, Tuple, Optional
import time
import random
import numpy as np
import pickle


def weighted_h(a: Tuple[int, int], b: Tuple[int, int], weights: List[List[int]]):
    return (
        (abs(a[0] - b[0]) + abs(a[1] - b[1])) // 5 +
        # Euclidean distance
        # np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 2 +
        (weights[a[1]][a[0]] + weights[b[1]][b[0]])  # * 5
    )


class Model:
    def __init__(
        self,
        matrix: List[List[int]],
        entry: Tuple[int, int],
        goal: Tuple[int, int],
        episodes: int,
        expected_plen: int,
        weights: List[List[int]]
    ):
        self.matrix = matrix
        self.entry = entry
        self.goal = goal
        self.reward = expected_plen

        self.width = len(matrix[0])
        self.height = len(matrix)
        self.weights = weights

        self.qtable = [
            [[0.0 for _ in range(4)] for _ in range(self.width)]
            for _ in range(self.height)
        ]

        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.valid_actions = [
            [[] for _ in range(self.width)] for _ in range(self.height)
        ]
        for y in range(self.height):
            for x in range(self.width):
                for i, (d_x, d_y) in enumerate(self.directions):
                    nx, ny = x + d_x, y + d_y
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and self.matrix[ny][nx] != 1
                    ):
                        self.valid_actions[y][x].append(i)

        self.episodes = episodes

        self.alpha_zero = 0.5
        self.alpha_min = 0.01
        self.alpha_decay = 0.001

        self.discount_factor = 0.99

        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min ** (1 / self.episodes)

    def learn(self) -> tuple[float, int]:
        startTime = time.time()

        nodesVisited = 0
        epsilon = 1
        max_steps = self.width * self.height

        for ep in range(self.episodes):
            position = self.entry
            steps = 0
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            alpha = max(self.alpha_min, self.alpha_zero / (1 + ep * self.alpha_decay))

            while position != self.goal:
                x, y = position
                nodesVisited += 1
                steps += 1
                valid_moves = self.valid_actions[y][x]

                if steps > max_steps or not valid_moves:
                    break

                if random.random() < epsilon:
                    move_idx = random.choice(valid_moves)
                else:
                    qvals = [self.qtable[y][x][i] for i in valid_moves]
                    max_q = max(qvals)
                    best_moves = [
                        valid_moves[i] for i, val in enumerate(qvals) if val == max_q
                    ]
                    move_idx = random.choice(best_moves)

                d_x, d_y = self.directions[move_idx]
                next_x, next_y = x + d_x, y + d_y

                old_dist = weighted_h((self.goal[0], self.goal[1]), (x, y), self.weights)
                new_dist = weighted_h((self.goal[0], self.goal[1]), (next_x,
                                                                     next_y), self.weights)

                if (next_x, next_y) == self.goal:
                    r = self.reward
                else:
                    r = old_dist - new_dist

                next_max = max(self.qtable[next_y][next_x])
                prev_value = self.qtable[y][x][move_idx]
                self.qtable[y][x][move_idx] += alpha * (
                    r + self.discount_factor * next_max - prev_value
                )

                position = (next_x, next_y)

            if not (ep + 1) % (self.episodes // 10):
                print(f"Training {(10 * (ep + 1) / (self.episodes // 10)):.2f}% done")

        return time.time() - startTime, nodesVisited

    def run(
        self,
    ) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
        startTime = time.time()
        position = self.entry
        path = [self.entry]
        steps = 0
        max_steps = self.reward * 1.5

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
            return None, time.time() - startTime, steps

        return path, time.time() - startTime, steps

    def serialize(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.qtable, f)

    def unserialize(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.qtable = pickle.load(f)
