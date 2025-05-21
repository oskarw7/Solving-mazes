import numpy as np
import pickle

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv


class HyperHeuristicController:
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        self.last_qtable_snapshot = None
        self.stable_counter = 0

    def select_parameters(self, episode: int, qtable) -> dict:
        # co 10 epizodow sprawdza stabilnosc qtable
        if episode % 10 == 0:
            if self.last_qtable_snapshot is not None:
                delta = np.abs(
                    np.array(self.last_qtable_snapshot) - np.array(qtable)).sum()
                if delta < 1e-3:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0
            self.last_qtable_snapshot = pickle.loads(
                pickle.dumps(qtable))

        # wybor strategii
        if episode < self.total_episodes * 0.3:
            strategy = {
                "epsilon_decay": 0.99,
                "alpha_decay": 0.0005,
                "reward_type": "dense"
            }
        elif self.stable_counter > 5:
            strategy = {
                "epsilon_decay": 0.97,
                "alpha_decay": 0.001,
                "reward_type": "potential"
            }
        else:
            strategy = {
                "epsilon_decay": 0.98,
                "alpha_decay": 0.0008,
                "reward_type": "mixed"
            }

        return strategy
