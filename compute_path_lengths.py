import json
import numpy as np
import matplotlib.pyplot as plt

_FILES = [
    # scene 04
    './scene04/bi_rrt_config_threshold_0.0_biasing10/scene04-bi_rrt_config_path.json',
    './scene04/bi_rrt_config_threshold_5.0_biasing10/scene04-bi_rrt_config_path.json',
    './scene04/bi_rrt_config_threshold_10.0_biasing10/scene04-bi_rrt_config_path.json',
    # scene 07
    './scene07/bi_rrt_config_threshold_0.0_biasing10/scene07-bi_rrt_config_path.json',
    './scene07/bi_rrt_config_threshold_5.0_biasing10/scene07-bi_rrt_config_path.json',
    './scene07/bi_rrt_config_threshold_10.0_biasing10/scene07-bi_rrt_config_path.json',
    # scene 08
    './scene08/bi_rrt_config_threshold_0.0_biasing10/scene08-bi_rrt_config_path.json',
    './scene08/bi_rrt_config_threshold_5.0_biasing10/scene08-bi_rrt_config_path.json',
    './scene08/bi_rrt_config_threshold_10.0_biasing10/scene08-bi_rrt_config_path.json',
    # scene 10
    # NOTE: threshold of 0 failed for scene 10 (collision-free grasp is not possible for this env)
    # so there is no file for it
    './scene10/bi_rrt_config_threshold_5.0_biasing10/scene10-bi_rrt_config_path.json',
    './scene10/bi_rrt_config_threshold_10.0_biasing10/scene10-bi_rrt_config_path.json',
    # scene 14
    './scene14/bi_rrt_config_threshold_0.0_biasing10/scene14-bi_rrt_config_path.json',
    './scene14/bi_rrt_config_threshold_5.0_biasing10/scene14-bi_rrt_config_path.json',
    './scene14/bi_rrt_config_threshold_10.0_biasing10/scene14-bi_rrt_config_path.json',
]

def compute_path_length(path_list):
    # Convert list of lists to a 2D NumPy array
    # shape will be (num_waypoints, num_joints)
    path_array = np.array(path_list)

    # Calculate the difference between consecutive waypoints
    # diffs[i] = path[i+1] - path[i]
    diffs = np.diff(path_array, axis=0)

    # Calculate the Euclidean distance for each step (L2 norm)
    # This squares the diffs, sums across joints, and takes the square root
    step_lengths = np.linalg.norm(diffs, axis=1)

    # Sum all steps to get total path length
    total_length = np.sum(step_lengths).item()
    return total_length


path_lengths = []
for f in _FILES:
    with open(f, 'r') as file:
        data = json.load(file)
        path = data['path']
        path_lengths.append(compute_path_length(path))
        if len(path_lengths) == 9:
            # add a placeholder value for the failed run (scene 10, threshold 0)
            path_lengths.append(0)

envs = ('scene 04', 'scene 07', 'scene 08', 'scene 10', 'scene 14')
threshold_path_lengths = {
    '0': path_lengths[::3],
    '5': path_lengths[1::3],
    '10': path_lengths[2::3],
}

x = np.arange(len(envs)) # label locations
width = 0.25    # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for threshold, length in threshold_path_lengths.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, length, width, label=threshold)
    ax.bar_label(rects, padding=3, fmt='%.2f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Joint-Space Path Length (rad)')
ax.set_title('Path Length by Collision Threshold')
ax.set_xticks(x + width, envs)
ax.legend(loc='upper right', ncols=3)

plt.show()
