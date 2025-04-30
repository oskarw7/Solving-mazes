from maze import *

def main():
    maze = Maze(100, 100)
    maze.generate()
    maze.draw()
    maze.saveMatrix('test_maze.pkl')

    loaded = loadMatrix('test_maze.pkl')
    newMaze = Maze(len(loaded[0]), len(loaded), loaded)
    maze.draw()


if __name__ == '__main__':
    main()