from typing import List, Tuple, Optional
import time


def dfsRecursive(matrix: List[List[int]], entry: Tuple[int, int], exit: Tuple[int, int]) -> [Optional[List[Tuple[int, int]]], float, int]:
    startTime = time.time()
    width = len(matrix[0])
    height = len(matrix)
    isVisited = [[False for _ in range(width)] for _ in range(height)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    path = []
    nodesVisited = 0

    def dfs(x: int, y: int) -> bool:
        nonlocal nodesVisited
        if matrix[y][x] == 1 or isVisited[y][x] or x < 0 or x >= width or y < 0 or y >= height or nodesVisited >= 1500:
            return False
        isVisited[y][x] = True
        nodesVisited += 1
        path.append((x, y))
        if (x, y) == exit:
            return True
        for directionX, directionY in directions:
            if dfs(x+directionX, y+directionY):
                return True
        path.pop()
        return False

    if dfs(entry[0], entry[1]):
        return path, time.time()-startTime, nodesVisited
    return None, time.time()-startTime, nodesVisited

def dfsIterative(matrix: List[List[int]], entry: Tuple[int, int], exit: Tuple[int, int]) -> [
    Optional[List[Tuple[int, int]]], float, int]:
    startTime = time.time()
    width = len(matrix[0])
    height = len(matrix)
    isVisited = [[False for _ in range(width)] for _ in range(height)]
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    nodesVisited = 0

    stack = [(entry[0], entry[1], [])]
    while stack:
        x, y, path = stack.pop()
        if matrix[y][x] == 1 or isVisited[y][x] or x < 0 or x >= width or y < 0 or y >= height:
            continue
        isVisited[y][x] = True
        nodesVisited += 1
        currentPath = path + [(x, y)]
        if (x, y) == exit:
            return currentPath, time.time() - startTime, nodesVisited
        for directionX, directionY in directions:
            stack.append((x + directionX, y + directionY, currentPath))

    return None, time.time() - startTime, nodesVisited