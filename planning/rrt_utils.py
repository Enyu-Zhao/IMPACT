import pybullet as p
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt


class VoxelGridNode:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.cost = 0 if parent is None else parent.cost
            

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]


def steer(from_node, to_node, step_size):
    direction = to_node - from_node
    length = np.linalg.norm(direction)
    direction = direction / length
    new_position = from_node + step_size * direction
    new_position = np.round(new_position).astype(int)
    return new_position


def get_line(start, end):
    start = np.array(start).astype(int)
    end = np.array(end).astype(int)
    points = []
    x1, y1, z1 = start
    x2, y2, z2 = end

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append([x1, y1, z1])
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append([x1, y1, z1])
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append([x1, y1, z1])
    return points


def is_free_node(position, voxel_grid, max_cost):
    x_dim, y_dim, z_dim = voxel_grid.shape
    x, y, z = position
    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
        return voxel_grid[x, y, z] <= max_cost
    return False


def is_free_edge(start, end, voxel_grid, max_cost):
    line = get_line(start, end)
    for point in line:
        if not is_free_node(point, voxel_grid, max_cost):
            return False
    return True


def grid_to_world(grid_pos, voxel_size, origin):
    return grid_pos * voxel_size + origin


def world_to_grid(world_pos, voxel_size, origin):
    grid_pos = (world_pos - origin) / voxel_size
    return np.round(grid_pos).astype(int)


def world_distance(point1, point2, voxel_size, origin, node_type):
    pos1 = point1.position if isinstance(point1, node_type) else point1
    pos2 = point2.position if isinstance(point2, node_type) else point2
    return np.linalg.norm(grid_to_world(pos1, voxel_size, origin) - grid_to_world(pos2, voxel_size, origin))


def evaluate_distance_to_obstacles(path, distance_transform):
    min_distance = float('inf')
    for pos in path:
        x, y, z = pos
        distance = distance_transform[x, y, z]
        if distance < min_distance:
            min_distance = distance
    return min_distance


def compute_distance_transform(voxel_grid, high_cost_threshold, voxel_size):
    obstacle_mask = voxel_grid > high_cost_threshold
    return distance_transform_edt(~obstacle_mask) * np.min(voxel_size)


def is_close_to(point1, point2, threshold=0.01):
    distance = np.linalg.norm(np.array(point1) - np.array(point2))
    return distance <= threshold


def draw_path(path):
    for i in range(len(path) - 1):
        p.addUserDebugLine(path[i][:3], path[i + 1][:3], lineColorRGB=[0, 1, 0], lineWidth=3.0)


def interpolate_path(path, steps):
    interpolated_path = []
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])
        for t in np.linspace(0, 1, steps, endpoint=False):
            interpolated_point = start * (1 - t) + end * t
            interpolated_path.append(interpolated_point.tolist())
    return interpolated_path


def get_initial_robot_state(robot):
    num_joints = p.getNumJoints(robot)
    initial_joint_states = [p.getJointState(robot, i) for i in range(num_joints)]
    initial_base_position, initial_base_orientation = p.getBasePositionAndOrientation(robot)
    initial_base_velocity, initial_base_angular_velocity = p.getBaseVelocity(robot)
    
    return {
        'joint_states': initial_joint_states,
        'base_position': initial_base_position,
        'base_orientation': initial_base_orientation,
        'base_velocity': initial_base_velocity,
        'base_angular_velocity': initial_base_angular_velocity,
    }


def smooth_path(path, polyorder=3, window_size=20):
    start_pos, end_pos = path[0], path[-1]

    savgol_polyorder = polyorder
    savgol_window_size = window_size

    savgol_window_size = len(path)
    savgol_polyorder = min(savgol_polyorder, savgol_window_size - 1)
    path = savgol_filter(path, savgol_window_size, savgol_polyorder, axis=0)
    path = path.tolist()
    path.insert(0, start_pos)
    path.append(end_pos)    
    return path
