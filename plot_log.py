import matplotlib.pyplot as plt
import numpy as np

# Parse the data
data = """
learned 397 5.39 34326 4124
a_star  397 0.04 32509 4293
learned 397 5.30 34093 4212
a_star  397 0.04 32119 4450
learned 397 5.31 33950 3732
a_star  397 0.04 32353 3833
learned 397 5.26 33914 4086
a_star  397 0.04 32092 4446
learned 397 5.47 34027 3800
a_star  397 0.04 32184 4416
learned 397 5.38 34223 4263
a_star  397 0.04 32557 4315
learned 397 5.38 34009 4223
a_star  397 0.04 32159 4798
learned 397 5.36 34109 3812
a_star  397 0.04 32034 4417
learned 397 5.35 33998 4187
a_star  397 0.04 32212 4465
learned 397 5.39 34137 3915
a_star  397 0.04 32109 4451
learned 397 5.35 34216 3977
a_star  397 0.04 32284 4346
learned 397 5.39 34248 3934
a_star  397 0.04 32495 4313
learned 397 5.40 34389 4243
a_star  397 0.04 32608 4139
learned 397 5.39 34071 4094
a_star  397 0.04 32195 4566
learned 397 5.43 34265 3713
a_star  397 0.04 32356 4165
learned 397 5.34 34221 3533
a_star  397 0.04 32342 4318
learned 397 5.32 33968 4157
a_star  397 0.04 32262 4604
learned 397 5.35 33920 4152
a_star  397 0.04 32103 4752
learned 397 5.42 34010 3454
a_star  397 0.04 32166 4340
learned 397 5.38 34235 4251
a_star  397 0.04 32463 4231
"""

# Parse the data
learned_nodes = []
learned_weights = []
astar_nodes = []
astar_weights = []

for line in data.strip().split('\n'):
    parts = line.split()
    algorithm = parts[0]
    nodes = int(parts[3])  # Third number (nodes visited)
    weight = int(parts[4])  # Fourth number (weight)

    if algorithm == 'learned':
        learned_nodes.append(nodes)
        learned_weights.append(weight)
    elif algorithm == 'a_star':
        astar_nodes.append(nodes)
        astar_weights.append(weight)

# Create scatter plot
plt.figure(figsize=(12, 8))

# Plot both datasets
plt.scatter(learned_nodes, learned_weights, c='red', alpha=0.7, s=60, label='Learned Heuristic', marker='o')
plt.scatter(astar_nodes, astar_weights, c='blue', alpha=0.7, s=60, label='A*', marker='s')

# Add labels and title
plt.xlabel('Nodes Visited', fontsize=12)
plt.ylabel('Weight', fontsize=12)
plt.title('Nodes Visited vs Weight: Learned Heuristic vs A*', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add some statistics as text
learned_avg_nodes = np.mean(learned_nodes)
astar_avg_nodes = np.mean(astar_nodes)
learned_avg_weight = np.mean(learned_weights)
astar_avg_weight = np.mean(astar_weights)

plt.text(0.02, 0.98, f'Learned Heuristic:\nAvg Nodes: {learned_avg_nodes:.0f}\nAvg Weight: {learned_avg_weight:.0f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))

plt.text(0.02, 0.78, f'A*:\nAvg Nodes: {astar_avg_nodes:.0f}\nAvg Weight: {astar_avg_weight:.0f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))

plt.tight_layout()
plt.savefig('nodes_vs_weight_comparison.png', dpi=300)
plt.show()

# Print summary statistics
print("Summary Statistics:")
print(f"Learned Heuristic - Avg Nodes: {learned_avg_nodes:.0f}, Avg Weight: {learned_avg_weight:.0f}")
print(f"A* - Avg Nodes: {astar_avg_nodes:.0f}, Avg Weight: {astar_avg_weight:.0f}")
print(f"Nodes Visited Difference: {learned_avg_nodes - astar_avg_nodes:.0f} (Learned visits more)")
print(f"Weight Difference: {astar_avg_weight - learned_avg_weight:.0f} (A* has higher weight)")
