from typing import List, Tuple, Optional


def dfsEntry(matrix: List[List[int]], entry: Tuple[int, int], exit: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    width = len(matrix[0])
    height = len(matrix)
    isVisited = [[False for _ in range(width)] for _ in range(height)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    path = []

    def dfs(x: int, y: int) -> bool:
        if matrix[y][x] == 1 or isVisited[y][x] or x < 0 or x >= width or y < 0 or y >= height:
            return False
        isVisited[y][x] = True
        path.append((x, y))
        if (x, y) == exit:
            return True
        for directionX, directionY in directions:
            if dfs(x+directionX, y+directionY):
                return True
        path.pop()
        return False

    if dfs(entry[0], entry[1]):
        return path
    return None
