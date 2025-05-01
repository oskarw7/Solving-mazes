from maze import *
from dfs import *

def main():
    maze = Maze(100, 100)
    maze.generate()
    maze.draw()

    start = (0, 1)
    goal = (maze.grid_w - 1, maze.grid_h - 2)
    path = dfsEntry(maze.grid, start, goal)
    maze.drawWithPath(path, "DFS")



if __name__ == '__main__':
    main()