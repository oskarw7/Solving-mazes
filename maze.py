import random
import matplotlib.pyplot as plt
import pickle

class DisjointSet:
    def __init__(self, width, height):
        self.parent = {(x, y): (x, y) for y in range(height) for x in range(width)}

    def find(self, cell):
        if self.parent[cell] != cell:
            self.parent[cell] = self.find(self.parent[cell])
        return self.parent[cell]

    def union(self, cell1, cell2):
        root1, root2 = self.find(cell1), self.find(cell2)
        if root1 != root2:
            self.parent[root2] = root1
            return True
        return False

class Maze:
    def __init__(self, width, height, grid=None):
        self.width = width
        self.height = height
        self.grid_w = 2 * width + 1
        self.grid_h = 2 * height + 1
        if grid is None:
            self.grid = [[1 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
            self.walls = self._generate_edges()
        else:
            self.grid = grid

    def _generate_edges(self):
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if x < self.width - 1:
                    walls.append(((x, y), (x + 1, y)))
                if y < self.height - 1:
                    walls.append(((x, y), (x, y + 1)))
        random.shuffle(walls)
        return walls

    def generate(self):
        ds = DisjointSet(self.width, self.height)
        for (x1, y1), (x2, y2) in self.walls:
            if ds.union((x1, y1), (x2, y2)):
                gx1, gy1 = 2 * x1 + 1, 2 * y1 + 1
                gx2, gy2 = 2 * x2 + 1, 2 * y2 + 1

                self.grid[gy1][gx1] = 0
                self.grid[gy2][gx2] = 0
                self.grid[(gy1 + gy2) // 2][(gx1 + gx2) // 2] = 0

                if random.random() < 0.1:
                    dx = gx2 - gx1
                    dy = gy2 - gy1
                    if dx == 0:
                        for offset in [-1, 1]:
                            if 0 <= gx1 + offset < self.grid_w:
                                self.grid[gy1][gx1 + offset] = 0
                                self.grid[gy2][gx2 + offset] = 0
                                self.grid[(gy1 + gy2) // 2][(gx1 + gx2) // 2 + offset] = 0
                    elif dy == 0:
                        for offset in [-1, 1]:
                            if 0 <= gy1 + offset < self.grid_h:
                                self.grid[gy1 + offset][gx1] = 0
                                self.grid[gy2 + offset][gx2] = 0
                                self.grid[(gy1 + gy2) // 2 + offset][(gx1 + gx2) // 2] = 0

        for x in range(self.grid_w):
            self.grid[0][x] = 1
            self.grid[self.grid_h - 1][x] = 1
        for y in range(self.grid_h):
            self.grid[y][0] = 1
            self.grid[y][self.grid_w - 1] = 1

        self.grid[1][0] = 0
        self.grid[self.grid_h - 2][self.grid_w - 1] = 0

    def draw(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.grid, cmap='Greys')
        ax.set_title("Labirynt z obramowaniem i jednym wejściem/wyjściem")
        ax.axis('off')
        plt.show()

    def saveMatrix(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump(self.grid, file)


def loadMatrix(filename: str) -> list[list[int]]:
    with open(filename, 'rb') as file:
        return pickle.load(file)
