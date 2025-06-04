from maze import Maze
from hh_learn_test import HeuristicLearner
from a_star import aStar
from typing import List, Tuple, Optional
import time


class LogFile:
    def __init__(self, filename: str):
        self.filename = filename

    def log(self, message: str):
        print(message)
        with open(self.filename, 'a') as f:
            f.write(message + '\n')

    def log_attempt(self):
        with open(self.filename, 'a') as f:
            f.write("\nAttempting to find a path...\n\n")


def learned_heuristic_astar(
    matrix: List[List[int]],
    entry: Tuple[int, int],
    exit: Tuple[int, int],
    heuristic_learner: HeuristicLearner = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
    import heapq as hq

    startTime = time.time()
    width = len(matrix[0])
    height = len(matrix)
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    nodesVisited = 0
    pathTracker = {}
    priorityQueue = []
    hq.heappush(
        priorityQueue, (0 + heuristic_learner.predict_heuristic(entry, exit), 0, entry)
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
            for i in range(len(path)):
                if heuristic_learner.weighed_grid is not None:
                    totalWeight += heuristic_learner.weighed_grid[path[i][1]][
                        path[i][0]
                    ]

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
                        gScores[adjacentPos]
                        + heuristic_learner.predict_heuristic(adjacentPos, exit),
                        gScores[adjacentPos],
                        adjacentPos,
                    ),
                )
    return None, time.time() - startTime, nodesVisited


if __name__ == "__main__":
    maze = Maze(500, 500)
    maze_type = "middle"
    maze.generate(mazeType=maze_type)

    log = LogFile("learned_heuristic_log.txt")

    maze.weighed_grid = maze.generate_weighed_grid_convolution()
    start_pos = (1, 1)
    goal_pos = (maze.grid_w - 2, maze.grid_h - 2)
    res = []

    heuristic_learner = HeuristicLearner(maze.grid, maze.weighed_grid)
    heuristic_learner.load_model("learned_heuristic_middle.pth")

    path, executionTime, nodesVisited, totalWeight = learned_heuristic_astar(
        maze.grid, start_pos, goal_pos, heuristic_learner
    )
    if path:
        log.log(
            f"Found path with {len(path)} steps using learned heuristic in {
                executionTime:.2f
            } seconds, visited {nodesVisited} nodes, total weight: {totalWeight}"
        )
        res.append(path)

    path, executionTime, nodesVisited, totalWeight = aStar(
        maze.grid, start_pos, goal_pos, maze.weighed_grid
    )
    if path:
        log.log(
            f"Found path with {len(path)} steps using A* in {executionTime:.2f} seconds, visited {nodesVisited} nodes, total weight: {totalWeight}"
        )
    res.append(path)

    maze.drawAll(res)
