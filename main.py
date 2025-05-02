from maze import *
from dfs import *
from a_star import *
from qlearning import *

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
    # aStarOnlyBenchmark()
    maze = Maze(100, 100)
    maze.generate()
    maze.draw()

    start = (0, 1)
    goal = (maze.grid_w-1, maze.grid_h-2)

    path, executionTime, nodesVisited = dfsIterative(maze.grid, start, goal)
    if path is not None:
        maze.drawWithPath(path, "DFS")
        print("DFS BENCHMARK:")
        print(f"\tPath length: {len(path)}")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("Path wasn't found")

    path, executionTime, nodesVisited = aStar(maze.grid, start, goal)
    if path is not None:
        maze.drawWithPath(path, "A*")
        print("A* BENCHMARK:")
        print(f"\tPath length: {len(path)}")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("Path wasn't found")

    path, executionTime, nodesVisited = qlearning(maze.grid, start, goal)
    if path is not None:
        maze.drawWithPath(path, "qlearning")
        print("QLEARNING BENCHMARK:")
        print(f"\tPath length: {len(path)}")
        print(f"\tExecution time: {executionTime}")
        print(f"\tTotal number of nodes visited: {nodesVisited}")
    else:
        print("Path wasn't found")


if __name__ == '__main__':
    main()