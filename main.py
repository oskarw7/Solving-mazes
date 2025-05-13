from maze import Maze, loadMatrix
from dfs import dfsIterative
from a_star import aStar
from qlearning import Model
import os
import random
import numpy as np


def dfsOnlyBenchmark():
    for i in range(1000):
        maze = Maze(100, 100)
        maze.generate()
        # maze.draw()

        entry = (0, 1)
        exit = (maze.grid_w - 1, maze.grid_h - 2)
        path, executionTime, nodesVisited = dfsIterative(maze.grid, entry, exit)
        if path is not None:
            # maze.drawWithPath(path, "DFS")
            print("DFS BENCHMARK:")
            print(f"\tPath length: {len(path)}")
            print(f"\tExecution time: {executionTime}")
            print(f"\tTotal number of nodes visited: {nodesVisited}")
        else:
            print("Path wasn't found")


def aStarOnlyBenchmark():
    for i in range(1000):
        maze = Maze(100, 100)
        maze.generate()
        # maze.draw()

        entry = (0, 1)
        exit = (maze.grid_w - 1, maze.grid_h - 2)
        path, executionTime, nodesVisited = aStar(maze.grid, entry, exit)
        if path is not None:
            # maze.drawWithPath(path, "A*")
            print("A* BENCHMARK:")
            print(f"\tPath length: {len(path)}")
            print(f"\tExecution time: {executionTime}")
            print(f"\tTotal number of nodes visited: {nodesVisited}")
        else:
            print("Path wasn't found")


def main():
    sed = 1
    random.seed(sed)
    np.random.seed(sed)
    res = []

    w = 1000
    h = 1000
    folder_name = f"{w}x{h}-s{sed}"
    maze = Maze(w, h)
    if os.path.isdir(folder_name):
        print("MAZE LOADED FROM FILE")
        maze.grid = loadMatrix(f"{folder_name}/matrix.pkl")
    else:
        maze.generate()
    # maze.draw()

    start = (0, 1)
    goal = (maze.grid_w - 1, maze.grid_h - 2)

    path, executionTime, nodesVisited = dfsIterative(maze.grid, start, goal)
    res.append(path)
    if path is not None:
        #maze.drawWithPath(path, "DFS")
        print("DFS BENCHMARK:")
        print(f"\tPath length: {len(path)}")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("Path wasn't found")

    path, executionTime, nodesVisited = aStar(maze.grid, start, goal)
    res.append(path)
    if path is not None:
        #maze.drawWithPath(path, "A*")
        print("A* BENCHMARK:")
        print(f"\tPath length: {len(path)}")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("Path wasn't found")

    qmodel = Model(maze.grid, start, goal, 5000, len(path))

    if not (os.path.isdir(folder_name) and os.path.isfile(f"{folder_name}/model.pkl")):
        executionTime, nodesVisited = qmodel.learn()
        print("QLEARNING TRAINING:")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("QLEARNING LOADED MODEL FROM FILE")
        qmodel.unserialize(f"{folder_name}/model.pkl")

    path, executionTime, nodesVisited = qmodel.run()
    res.append(path)
    if path is not None:
        #maze.drawWithPath(path, "qlearning")
        print("QLEARNING RESULT:")
        print(f"\tPath length: {len(path)}")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("Path wasn't found")

    maze.drawAll(res)

    os.makedirs(folder_name, exist_ok=True)
    maze.saveMatrix(f"{folder_name}/matrix.pkl")
    qmodel.serialize(f"{folder_name}/model.pkl")

    print(f"saved maze and model to directory {folder_name}/")


if __name__ == "__main__":
    main()
