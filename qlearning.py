from typing import List, Tuple, Optional
import time
import random
import numpy as np
from a_star import h  # assuming you have a heuristic function

import pickle

def qlearning(matrix: List[List[int]], entry: Tuple[int, int], exit: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
    startTime = time.time()

    width = len(matrix[0])
    height = len(matrix)

    qtable = [[[0.0 for _ in range(4)] for _ in range(width)] for _ in range(height)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    nodesVisited = 0

    learning_rate = 0.1
    discount_factor = 0.99
    episodes = 5000
    reward = 1000
    max_steps = width * height * 2

    for ep in range(episodes):
        position = entry
        steps = 0
        epsilon = max(0.01, 1.0 - ep / episodes)

        while position != exit:
            x, y = position
            nodesVisited += 1
            steps += 1

            if steps > max_steps:
                # print(f"Episode {ep + 1}: Max steps reached")
                break

            if random.random() < epsilon:
                move_idx = random.randint(0, 3)
            else:
                qvals = qtable[y][x]
                max_q = max(qvals)
                best_moves = [i for i, val in enumerate(qvals) if val == max_q]
                move_idx = random.choice(best_moves)

            d_x, d_y = directions[move_idx]
            next_x = x + d_x
            next_y = y + d_y

            old_dist = abs(exit[0] - x) + abs(exit[1] - y)
            new_dist = abs(exit[0] - next_x) + abs(exit[1] - next_y)

            if not (0 <= next_x < width and 0 <= next_y < height):
                r = -10
                next_x, next_y = x, y
            elif matrix[next_y][next_x] == 1:
                r = -10
                next_x, next_y = x, y
            elif (next_x, next_y) == exit:
                r = reward
            else:
                r = -1 + (old_dist - new_dist)

            next_max = max(qtable[next_y][next_x])
            prev_value = qtable[y][x][move_idx]
            qtable[y][x][move_idx] += learning_rate * (r + discount_factor * next_max - prev_value)

            position = (next_x, next_y)

        # print(f"Episode {ep + 1}/{episodes} completed")

    with open("qlearned.pkl", 'wb') as file:
        pickle.dump(qtable, file)

    # extract path from trained Q-table
    position = entry
    path = [entry]
    steps = 0

    while position != exit and steps < max_steps:
        x, y = position
        steps += 1

        move_idx = int(np.argmax(qtable[y][x]))
        d_x, d_y = directions[move_idx]
        next_x, next_y = x + d_x, y + d_y

        if not (0 <= next_x < width and 0 <= next_y < height):
            break
        if matrix[next_y][next_x] != 0:
            break

        position = (next_x, next_y)
        path.append(position)

    if position != exit:
        return None, time.time() - startTime, nodesVisited

    return path, time.time() - startTime, nodesVisited
