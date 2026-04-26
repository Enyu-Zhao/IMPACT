import pybullet as p
import numpy as np

from planning.rrt_utils import world_to_grid, distance, compute_distance_transform


class ConfigNode:
    def __init__(self, joint_positions, parent=None):
        self.position = np.array(joint_positions, dtype=float)
        self.parent = parent
        self.cost = 0 if parent is None else parent.cost


def config_to_ee_pos(robot, joint_positions, ee_link_index):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)
    return np.array(p.getLinkState(robot, ee_link_index)[4])


def config_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin):
    # Key function for getting the cost of a configuration based on the end-effector position in the voxel grid
    ee_pos = config_to_ee_pos(robot, joint_positions, ee_link_index)
    x, y, z = world_to_grid(ee_pos, voxel_size, origin)
    x_dim, y_dim, z_dim = voxel_grid.shape
    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
        return voxel_grid[x, y, z]
    return float('inf')


def robot_max_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)
    x_dim, y_dim, z_dim = voxel_grid.shape
    max_cost = 0
    for link_idx in range(ee_link_index + 1):
        link_pos = np.array(p.getLinkState(robot, link_idx)[4])
        x, y, z = world_to_grid(link_pos, voxel_size, origin)
        if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
            cost = voxel_grid[x, y, z]
            if cost > max_cost:
                max_cost = cost
    return max_cost


def is_free_config(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, max_cost, whole_robot=False):
    if whole_robot:
        cost = robot_max_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin)
    else:
        cost = config_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin)
    return cost <= max_cost


def is_free_config_edge(config1, config2, robot, ee_link_index, voxel_grid, voxel_size, origin, max_cost, steps=10, whole_robot=False):
    for i in range(1, steps + 1):
        t = i / steps
        interp = config1 + t * (config2 - config1)
        if not is_free_config(robot, interp, ee_link_index, voxel_grid, voxel_size, origin, max_cost, whole_robot):
            return False
    return True


def is_in_unsafe_contact(robot, cost_dict, max_cost, ignored_body_ids=None, ignored_robot_link_ids=None):
    """Return True when the robot touches a body whose cost exceeds max_cost.

    Bodies not in cost_dict are treated as fully blocking. Contacts with the
    target are allowed when the target is marked with cost -1, or when its body
    id is included in ignored_body_ids. PyBullet uses robot link index -1 for
    the base link.
    """
    ignored_body_ids = set() if ignored_body_ids is None else set(ignored_body_ids)
    ignored_robot_link_ids = set() if ignored_robot_link_ids is None else set(ignored_robot_link_ids)

    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot)
    for contact in contacts:
        body_b = contact[2]
        robot_link = contact[3]
        if robot_link in ignored_robot_link_ids or body_b == robot or body_b in ignored_body_ids:
            continue
        cost = cost_dict.get(body_b, float('inf'))
        if cost == -1:
            continue
        if cost > max_cost:
            return True
    return False


def config_has_unsafe_contact(robot, joint_positions, cost_dict, max_cost, ignored_body_ids=None, ignored_robot_link_ids=None):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)
    return is_in_unsafe_contact(robot, cost_dict, max_cost, ignored_body_ids, ignored_robot_link_ids)


def has_blocking_contact(robot, cost_dict=None, max_contact_cost=9, ignored_body_ids=None, ignored_robot_link_ids=None):
    ignored_body_ids = set() if ignored_body_ids is None else set(ignored_body_ids)
    ignored_robot_link_ids = set() if ignored_robot_link_ids is None else set(ignored_robot_link_ids)
    if cost_dict is not None:
        return is_in_unsafe_contact(robot, cost_dict, max_contact_cost, ignored_body_ids=ignored_body_ids, ignored_robot_link_ids=ignored_robot_link_ids)

    p.performCollisionDetection()
    return any(contact[3] not in ignored_robot_link_ids and contact[2] not in ignored_body_ids for contact in p.getContactPoints(bodyA=robot))


def is_valid_config(robot, joint_positions, cost_dict=None, max_contact_cost=9, ignored_body_ids=None, ignored_robot_link_ids=None, ee_link_index=None, voxel_grid=None, voxel_size=None, origin=None, max_voxel_cost=None, whole_robot=False):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)

    if voxel_grid is not None:
        if ee_link_index is None or voxel_size is None or origin is None or max_voxel_cost is None:
            raise ValueError("ee_link_index, voxel_size, origin, and max_voxel_cost are required when voxel_grid is provided.")
        if not is_free_config(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, max_voxel_cost, whole_robot=whole_robot):
            return False

    return not has_blocking_contact(robot, cost_dict=cost_dict, max_contact_cost=max_contact_cost, ignored_body_ids=ignored_body_ids, ignored_robot_link_ids=ignored_robot_link_ids)


def config_intersects_target(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin):
    ee_pos = config_to_ee_pos(robot, joint_positions, ee_link_index)
    x, y, z = world_to_grid(ee_pos, voxel_size, origin)
    x_dim, y_dim, z_dim = voxel_grid.shape
    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
        return voxel_grid[x, y, z] == -1
    return False


def reconstruct_config_path(node):
    path = []
    while node:
        path.append(node.position.tolist())
        node = node.parent
    return path[::-1]


def steer_config(from_config, to_config, step_size):
    direction = to_config - from_config
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return from_config.copy()
    return from_config + step_size * (direction / length)


def evaluate_config_path_cost(path, robot, ee_link_index, voxel_grid, voxel_size, origin, whole_robot=False):
    total_cost = 0
    for joint_positions in path:
        if whole_robot:
            cost = robot_max_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin)
        else:
            cost = config_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin)
        if cost != float('inf'):
            total_cost += cost
    return total_cost


def evaluate_config_distance_to_obstacles(path, robot, ee_link_index, distance_transform, voxel_size, origin, whole_robot=False):
    min_distance = float('inf')
    x_dim, y_dim, z_dim = distance_transform.shape
    for joint_positions in path:
        if whole_robot:
            for i, pos in enumerate(joint_positions):
                p.resetJointState(robot, i, pos)
            link_positions = [np.array(p.getLinkState(robot, idx)[4]) for idx in range(ee_link_index + 1)]
        else:
            link_positions = [config_to_ee_pos(robot, joint_positions, ee_link_index)]

        for link_pos in link_positions:
            x, y, z = world_to_grid(link_pos, voxel_size, origin)
            if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
                d = distance_transform[x, y, z]
                if d < min_distance:
                    min_distance = d
    return min_distance
