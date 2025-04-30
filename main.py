from maze import *

def main():
    maze = Maze(20, 20)
    maze.generate()
    maze.draw()
    maze.saveMatrix('test_maze.pkl')

    loaded = loadMatrix('test_maze.pkl')
    print(loaded)
    newMaze = Maze(len(loaded[0]), len(loaded), loaded)
    maze.draw()


if __name__ == '__main__':
    main()