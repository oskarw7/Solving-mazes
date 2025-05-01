from maze import *
from dfs import *

def dfsOnlyBenchmark():
    for i in range(1000):
        maze = Maze(100, 100)
        maze.generate()
        # maze.draw()

        start = (0, 1)
        goal = (maze.grid_w - 1, maze.grid_h - 2)
        path, executionTime, nodesVisited = dfsIterative(maze.grid, start, goal)
        if path is not None:
            # maze.drawWithPath(path, "DFS")
            print("DFS BENCHMARK:")
            print(f"\tPath length: {len(path)}")
            print(f"\tExecution time: {executionTime}")
            print(f"\tTotal number of nodes visited: {nodesVisited}")
        else:
            print("Number of nodes visited exceeded 1500")

def main():
    maze = Maze(1000, 1000)
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
        print("Number of nodes visited exceeded 1500")

if __name__ == '__main__':
    main()