import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import random
from maze import Maze
from qlearning import Model


class HeuristicFeatureExtractor:
    """
    Extracts features from maze states that can be used to learn heuristics
    """

    def __init__(
        self, maze_grid: List[List[int]], weighed_grid: List[List[int]] = None
    ):
        self.grid = np.array(maze_grid)
        self.height, self.width = self.grid.shape
        if weighed_grid is not None:
            self.weighed_grid = np.array(weighed_grid)
        else:
            self.weighed_grid = np.ones_like(self.grid, dtype=np.int8)

    def extract_features(
        self, pos: Tuple[int, int], goal: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract features for heuristic learning
        Returns feature vector of size ~20-30
        """
        x, y = pos
        gx, gy = goal

        features = []

        # dystanse
        features.extend(
            [
                abs(x - gx),  # Manhattan distance X
                abs(y - gy),  # Manhattan distance Y
                # Chebyshev distance
                # max(abs(x - gx), abs(y - gy)),
            ]
        )

        # kierunek do celu (trenowanie jest od losowych pozycji do losowych pozycji)
        dx, dy = gx - x, gy - y
        features.extend(
            [
                1 if dx > 0 else 0,
                1 if dx < 0 else 0,
                1 if dy > 0 else 0,
                1 if dy < 0 else 0,
            ]
        )

        # WEIGHTED GRID FEATURES - Key addition for crowding penalties
        if self.weighed_grid is not None:
            # Current position weight (crowding penalty)
            current_weight = (
                self.weighed_grid[y][x]
                if 0 <= x < self.width and 0 <= y < self.height
                else 0
            )
            features.append(current_weight)

            # Average weight in surrounding area (local crowding)
            weight_sums = []
            for radius in [1, 2, 3]:
                total_weight = 0
                cells = 0
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < self.width
                            and 0 <= ny < self.height
                            and self.grid[ny][nx] == 0
                        ):
                            total_weight += self.weighed_grid[ny][nx]
                            cells += 1
                avg_weight = total_weight / cells if cells > 0 else 0
                weight_sums.append(avg_weight)
            features.extend(weight_sums)

            # Weight gradient towards goal (is path getting more or less crowded?)
            weight_gradient = self._calculate_weight_gradient(pos, goal)
            features.append(weight_gradient)

            # Minimum weight path estimate (rough estimate of best possible route)
            min_weight_estimate = self._estimate_min_weight_path(pos, goal)
            features.append(min_weight_estimate)

        # # sprawdzenie czy sciezka jest zablokowana
        # direct_blocked = self._count_walls_in_line(pos, goal)
        # features.append(direct_blocked)
        #
        # # wykrywanie korytarzy
        # corridor_score = self._detect_corridor(pos)
        # features.append(corridor_score)
        #
        # # wykrywanie slepych uliczek
        # deadend_score = self._detect_nearby_deadends(pos)
        # features.append(deadend_score)
        #
        # # alternatywne sciezki
        # alt_paths = self._count_alternative_paths(pos, goal)
        # features.append(alt_paths)
        #
        # # sprawdzanie widocznosci celu
        # visibility = self._goal_visibility(pos, goal)
        # features.extend(visibility)

        return np.array(features, dtype=np.float32)

    def _calculate_weight_gradient(
        self, pos: Tuple[int, int], goal: Tuple[int, int]
    ) -> float:
        """
        Calculate if moving towards goal increases or decreases crowding
        Positive value means getting more crowded, negative means less crowded
        """
        if self.weighed_grid is None:
            return 0.0

        x, y = pos
        gx, gy = goal

        # Sample points along the direct line to goal
        steps = max(abs(gx - x), abs(gy - y))
        if steps == 0:
            return 0.0

        weights_along_path = []
        for i in range(min(steps, 10)):  # Sample up to 10 points
            progress = (i + 1) / min(steps, 10)
            sample_x = int(x + (gx - x) * progress)
            sample_y = int(y + (gy - y) * progress)

            if 0 <= sample_x < self.width and 0 <= sample_y < self.height:
                if self.grid[sample_y][sample_x] == 0:  # Not a wall
                    weights_along_path.append(self.weighed_grid[sample_y][sample_x])

        if len(weights_along_path) < 2:
            return 0.0

        # Calculate trend: positive = getting more crowded, negative = less crowded
        start_avg = np.mean(weights_along_path[: len(weights_along_path) // 2])
        end_avg = np.mean(weights_along_path[len(weights_along_path) // 2:])

        return end_avg - start_avg

    def _estimate_min_weight_path(
        self, pos: Tuple[int, int], goal: Tuple[int, int]
    ) -> float:
        """
        Rough estimate of minimum weight path to goal
        """
        if self.weighed_grid is None:
            return 0.0

        x, y = pos
        gx, gy = goal

        # Simple sampling approach: check a few potential paths
        min_avg_weight = float("inf")

        # Direct path
        direct_weights = []
        steps = max(abs(gx - x), abs(gy - y))
        for i in range(steps + 1):
            if steps == 0:
                sample_x, sample_y = x, y
            else:
                sample_x = int(x + (gx - x) * i / steps)
                sample_y = int(y + (gy - y) * i / steps)

            if 0 <= sample_x < self.width and 0 <= sample_y < self.height:
                if self.grid[sample_y][sample_x] == 0:
                    direct_weights.append(self.weighed_grid[sample_y][sample_x])

        if direct_weights:
            min_avg_weight = min(min_avg_weight, np.mean(direct_weights))

        # Sample a few alternative paths (simplified)
        for offset in [-1, 0, 1]:
            alt_weights = []
            mid_x = (x + gx) // 2 + offset
            mid_y = (y + gy) // 2 + offset

            if 0 <= mid_x < self.width and 0 <= mid_y < self.height:
                if self.grid[mid_y][mid_x] == 0:
                    # Path through midpoint
                    for px, py in [(x, y), (mid_x, mid_y), (gx, gy)]:
                        if 0 <= px < self.width and 0 <= py < self.height:
                            if self.grid[py][px] == 0:
                                alt_weights.append(self.weighed_grid[py][px])

                    if alt_weights:
                        min_avg_weight = min(min_avg_weight, np.mean(alt_weights))

        return min_avg_weight if min_avg_weight != float("inf") else 0.0

    def _count_walls_in_line(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> float:
        """Count walls between start and end positions"""
        x1, y1 = start
        x2, y2 = end

        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return 0

        walls = 0
        for i in range(1, steps):
            x = int(x1 + (x2 - x1) * i / steps)
            y = int(y1 + (y2 - y1) * i / steps)
            if 0 <= x < self.width and 0 <= y < self.height:
                walls += self.grid[y][x]

        return walls / steps

    def _detect_corridor(self, pos: Tuple[int, int]) -> float:
        """Detect if position is in a corridor"""
        x, y = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        open_dirs = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny][nx] == 0:
                    open_dirs += 1

        return 1.0 if open_dirs <= 2 else 0.0

    def _detect_nearby_deadends(self, pos: Tuple[int, int]) -> float:
        """Detect dead-ends in nearby area"""
        x, y = pos
        deadends = 0

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx] == 0:  # Not a wall
                        # Count adjacent walls
                        wall_count = 0
                        for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nnx, nny = nx + ddx, ny + ddy
                            if (
                                0 <= nnx < self.width
                                and 0 <= nny < self.height
                                and self.grid[nny][nnx] == 1
                            ):
                                wall_count += 1

                        if wall_count >= 3:  # Dead end
                            deadends += 1

        return deadends / 25.0  # normalizacja

    def _count_alternative_paths(
        self, pos: Tuple[int, int], goal: Tuple[int, int]
    ) -> float:
        """Estimate number of alternative paths"""
        x, y = pos
        gx, gy = goal

        # Count open cells in general direction of goal
        alt_paths = 0
        search_radius = 3

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx] == 0:
                        # is this cell is roughly in direction of goal
                        if (
                            (gx > x and nx > x)
                            or (gx < x and nx < x)
                            or (gy > y and ny > y)
                            or (gy < y and ny < y)
                        ):
                            alt_paths += 1

        return alt_paths / ((2 * search_radius + 1) ** 2)

    def _goal_visibility(
        self, pos: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[float]:
        """Check if goal is visible in cardinal directions"""
        x, y = pos
        gx, gy = goal

        visibility = [0.0, 0.0, 0.0, 0.0]  # right, down, left, up
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for i, (dx, dy) in enumerate(directions):
            step = 1
            while True:
                nx, ny = x + dx * step, y + dy * step
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    break
                if self.grid[ny][nx] == 1:  # sciana
                    break
                if (nx, ny) == (gx, gy):  # cel widoczny
                    visibility[i] = 1.0
                    break
                step += 1

        return visibility


class NeuralHeuristic(nn.Module):
    """
    Neural network that learns to predict heuristic values
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.1)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HeuristicLearner:
    """
    Main class for learning heuristics from maze solving experience
    """

    def __init__(
        self,
        maze_grid: List[List[int]],
        weighed_grid: List[List[int]] = None,
        device="cpu",
    ):
        self.maze_grid = maze_grid
        self.weighed_grid = weighed_grid
        self.feature_extractor = HeuristicFeatureExtractor(maze_grid)
        self.device = device

        sample_features = self.feature_extractor.extract_features((1, 1), (2, 2))
        self.feature_size = len(sample_features)

        self.heuristic_net = NeuralHeuristic(self.feature_size).to(device)
        self.optimizer = optim.Adam(self.heuristic_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.training_data = []

    def collect_training_data_from_astar(
        self, start: Tuple[int, int], goal: Tuple[int, int], path: List[Tuple[int, int]]
    ):
        """
        Collect training data from A* solutions with weighed penalties
        Target heuristic = remaining distance + accumulated weight penalties
        """
        for i, pos in enumerate(path):
            remaining_steps = len(path) - 1 - i

            # Calculate weight penalty for remaining path
            weight_penalty = 0.0
            if self.weighed_grid is not None:
                for j in range(i, len(path)):
                    px, py = path[j]
                    if 0 <= px < len(self.weighed_grid[0]) and 0 <= py < len(
                        self.weighed_grid
                    ):
                        # Use the maze's reward system (negative for higher weights)
                        weight_penalty += self.maze.get_reward((px, py))

            features = self.feature_extractor.extract_features(pos, goal)
            # Combine distance and weight penalty
            target_heuristic = (
                remaining_steps - weight_penalty
            )  # Negative weight_penalty makes this larger
            self.training_data.append((features, target_heuristic))

    def collect_training_data_from_qlearning(
        self, q_table: List[List[List[float]]], goal: Tuple[int, int]
    ):
        """
        Collect training data from Q-learning results
        Use Q-values as heuristic targets
        """
        height, width = len(q_table), len(q_table[0])

        for y in range(height):
            for x in range(width):
                if self.maze_grid[y][x] == 0:
                    features = self.feature_extractor.extract_features((x, y), goal)
                    # Use max Q-value as heuristic estimate
                    heuristic_value = max(q_table[y][x]) if q_table[y][x] else 0

                    weighed_penalty = 0.0
                    if self.weighed_grid is not None:
                        # Apply weighed penalty based on the grid
                        weighed_penalty = self.weighed_grid[y][x]

                    heuristic_value -= weighed_penalty  # Adjust heuristic by penalty
                    self.training_data.append((features, heuristic_value))

    def train(self, epochs: int = 1000, batch_size: int = 32):
        """
        Train the heuristic neural network (optimized version)
        """
        if not self.training_data:
            print("No training data available!")
            return

        print(f"Training on {len(self.training_data)} samples...")

        features_list = [item[0] for item in self.training_data]
        targets_list = [item[1] for item in self.training_data]

        # Convert to numpy arrays once
        all_features = np.array(features_list, dtype=np.float32)
        all_targets = np.array(targets_list, dtype=np.float32).reshape(-1, 1)

        for epoch in range(epochs):
            indices = list(range(len(self.training_data)))
            random.shuffle(indices)

            total_loss = 0
            batches = 0

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i: i + batch_size]

                # indeksowanie numpy
                batch_features = all_features[batch_indices]
                batch_targets = all_targets[batch_indices]

                features = torch.from_numpy(batch_features).to(self.device)
                targets = torch.from_numpy(batch_targets).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.heuristic_net(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            if (epoch + 1) % (epochs//5) == 0:
                avg_loss = total_loss / batches if batches > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def predict_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Predict heuristic value for given position and goal
        """
        features = self.feature_extractor.extract_features(pos, goal)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heuristic = self.heuristic_net(features_tensor).item()

        return max(0, heuristic)

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {
                "model_state_dict": self.heuristic_net.state_dict(),
                "feature_size": self.feature_size,
                "training_data_size": len(self.training_data),
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.heuristic_net.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model trained on {checkpoint['training_data_size']} samples")


# full wygenerowane, blueprint do naszej implementacji
def learned_heuristic_astar(
    maze: Maze,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic_learner: HeuristicLearner,
):
    """
    A* implementation using learned heuristic
    """
    from heapq import heappush, heappop

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_learner.predict_heuristic(start, goal)}

    visited = set()

    while open_set:
        current = heappop(open_set)[1]

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        x, y = current
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor

            if (
                0 <= nx < maze.grid_w
                and 0 <= ny < maze.grid_h
                and maze.grid[ny][nx] == 0
            ):
                # Calculate step cost including weight penalty
                step_cost = 1.0
                if hasattr(maze, "weighed_grid") and maze.weighed_grid is not None:
                    try:
                        # Add weight penalty to step cost
                        weight_penalty = maze.get_reward(neighbor)
                        step_cost += weight_penalty
                    except:
                        pass
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = (
                        tentative_g
                        + heuristic_learner.predict_heuristic(neighbor, goal)
                    )
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None


def integrate():
    # TODO:  wytrenowac modele na bazie mazeType'ow, ewentualnie tweak
    # parametrow do gestosci labiryntu
    maze_type_to_learn = "middle"
    maze = Maze(50, 50)
    maze.generate(mazeType=maze_type_to_learn)
    maze.weighed_grid = maze.generate_weighed_grid_convolution()

    heuristic_learner = HeuristicLearner(maze.grid)

    # Metoda 1: Zbieranie danych z Q-learningu
    start_pos = (1, 1)
    goal_pos = (maze.grid_w - 2, maze.grid_h - 2)

    model = Model(maze.grid, start_pos, goal_pos, 1000,
                  100, maze.weighed_grid)
    model.learn()

    heuristic_learner.collect_training_data_from_qlearning(model.qtable, goal_pos)

    # Metoda 2: Nauka z losowych startów i celów
    for _ in range(100):
        while True:
            sx, sy = (
                random.randint(1, maze.grid_w - 2),
                random.randint(1, maze.grid_h - 2),
            )
            print("Trying start position:", (sx, sy))
            if maze.grid[sy][sx] == 0:
                print("Found valid start position:", (sx, sy))
                break
        while True:
            gx, gy = (
                random.randint(1, maze.grid_w - 2),
                random.randint(1, maze.grid_h - 2),
            )
            print("Trying goal position:", (gx, gy))
            if maze.grid[gy][gx] == 0:
                print("Found valid goal position:", (gx, gy))
                break

        print(f"Collecting data for start: {(sx, sy)}, goal: {(gx, gy)}")
        temp_model = Model(
            maze.grid, (sx, sy), (gx, gy), 1000, 100,
            maze.weighed_grid,
        )
        temp_model.learn()
        heuristic_learner.collect_training_data_from_qlearning(
            temp_model.qtable, (gx, gy)
        )
        print(f"Collected data for start: {(sx, sy)}, goal: {(gx, gy)}")

    epochs = 50
    heuristic_learner.train(epochs=epochs)

    # zapis modelu
    heuristic_learner.save_model(f"learned_heuristic_{maze_type_to_learn}.pth")

    # Test
    path = learned_heuristic_astar(maze, start_pos, goal_pos, heuristic_learner)
    if path:
        print(f"Found path with {len(path)} steps using learned heuristic")
        maze.drawWithPath(path, "Learned Heuristic A*")

    return heuristic_learner


if __name__ == "__main__":
    heuristic_learner = integrate()
