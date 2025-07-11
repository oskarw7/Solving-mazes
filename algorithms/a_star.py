import heapq as hq
import time
from typing import List, Tuple, Optional


# Manhattan distance
def manhattan_h(a: Tuple[int, int], b: Tuple[int, int]):
    # return abs(a[0] - b[0]) + abs(a[1] - b[1])
    # Test with Euclidean distance
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def aStar(
    matrix: List[List[int]], entry: Tuple[int, int], exit: Tuple[int, int],
    weights: List[List[int]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
    startTime = time.time()
    width = len(matrix[0])
    height = len(matrix)
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    nodesVisited = 0
    pathTracker = {}
    priorityQueue = []
    hq.heappush(
        priorityQueue, (0 + manhattan_h(entry, exit), 0, entry)
    )  # stored like: (fScore, gScore, position)
    gScores = {entry: 0}

    while priorityQueue:
        fScore, gScore, position = hq.heappop(priorityQueue)
        nodesVisited += 1
        if position == exit:
            path = [position]
            while position in pathTracker:
                position = pathTracker[position]
                path.insert(0, position)
            totalWeight = 0
            if weights is not None:
                for i in range(len(path)):
                    totalWeight += weights[path[i][1]][path[i][0]]
            return path, time.time() - startTime, nodesVisited, totalWeight
        for directionX, directionY in directions:
            x, y = position[0] + directionX, position[1] + directionY
            if matrix[y][x] == 1 or 0 > x >= width or 0 > y >= height:
                continue
            adjacentPos = (x, y)
            if adjacentPos not in gScores or gScore + 1 < gScores[adjacentPos]:
                gScores[adjacentPos] = gScore + 1
                pathTracker[adjacentPos] = position
                hq.heappush(
                    priorityQueue,
                    (
                        gScores[adjacentPos] + manhattan_h(adjacentPos, exit),
                        gScores[adjacentPos],
                        adjacentPos,
                    ),
                )
    return None, time.time() - startTime, nodesVisited
