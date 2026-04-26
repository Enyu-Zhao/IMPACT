import pybullet as p
import numpy as np

from planning.rrt_utils import world_to_grid


def ee_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin):
    # Key function for getting the cost of a configuration based on the end-effector position in the voxel grid
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)
    ee_pos = np.array(p.getLinkState(robot, ee_link_index)[4])
    x, y, z = world_to_grid(ee_pos, voxel_size, origin)
    x_dim, y_dim, z_dim = voxel_grid.shape
    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
        return voxel_grid[x, y, z]
    return float('inf')


def robot_max_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, use_all_occupied_voxels=False):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)

    # Calculate the cost for the whole robot.
    if use_all_occupied_voxels:
        costs = robot_occupied_voxel_costs(
            robot,
            joint_positions,
            voxel_grid,
            voxel_size,
            origin,
            link_indices=range(ee_link_index + 1),
        )
        return max(costs) if costs else float('inf')

    # Simplified version that only considers the voxel cost at each link's frame position.
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


def robot_occupied_voxel_costs(robot, joint_positions, voxel_grid, voxel_size, origin, link_indices=None):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)

    if link_indices is None:
        link_indices = range(p.getNumJoints(robot))

    costs = []
    max_valid = np.array(voxel_grid.shape) - 1
    for link_idx in link_indices:
        if link_idx == -1:
            continue

        aabb_min, aabb_max = p.getAABB(robot, link_idx)
        min_idx = np.floor((np.array(aabb_min) - origin) / voxel_size).astype(int)
        max_idx = np.floor((np.array(aabb_max) - origin) / voxel_size).astype(int)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, max_valid)
        if np.any(min_idx > max_idx):
            continue

        for x in range(min_idx[0], max_idx[0] + 1):
            for y in range(min_idx[1], max_idx[1] + 1):
                for z in range(min_idx[2], max_idx[2] + 1):
                    costs.append(voxel_grid[x, y, z])
    return costs







def is_free_config(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, max_cost, whole_robot=False, use_all_occupied_voxels=False):
    if whole_robot:
        cost = robot_max_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, use_all_occupied_voxels=use_all_occupied_voxels)
    else:
        cost = ee_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin)
    return cost <= max_cost



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

# Main function for RRT-based planning to call from both bi_rrt.py and impact_planning.py, which includes the common code for generating valid IK goal configurations and checking config validity.
def is_valid_config(
    robot, 
    joint_positions, 
    cost_dict=None, 
    max_cost=9, 
    ignored_body_ids=None, 
    ignored_robot_link_ids=None, 
    ee_link_index=None, 
    voxel_grid=None, 
    voxel_size=None, 
    origin=None, 
    whole_robot=False, 
    use_all_occupied_voxels=False):
    """Check whether a robot joint configuration is valid.

    The check has two parts:
    1. Voxel-cost checking, enabled when voxel_grid is provided.
    2. PyBullet contact checking, always performed.

    Args:
        robot: PyBullet body id for the robot.
        joint_positions: Joint values for the robot arm configuration.
        cost_dict: Optional mapping from PyBullet body id to object cost.
            Missing body ids are treated as infinite cost, so contact with them
            is blocking. A cost of -1 is treated as the target and allowed.
        max_cost: Maximum allowed cost for both voxel cells and contacted
            objects. Costs above this value make the configuration invalid.
        ignored_body_ids: Body ids to ignore during contact checking, commonly
            the target object id.
        ignored_robot_link_ids: Robot link ids to ignore during contact
            checking. PyBullet uses -1 for the base link.
        ee_link_index: End-effector link index. Required if voxel_grid is used.
        voxel_grid: Optional cost grid used for voxel-cost checking.
        voxel_size: World-space size of each voxel. Required if voxel_grid is
            used.
        origin: World-space origin of the voxel grid. Required if voxel_grid is
            used.
        whole_robot: If False, voxel checking only uses the end-effector voxel.
            If True, voxel checking uses robot links up to ee_link_index.
        use_all_occupied_voxels: Only used when whole_robot is True. If False,
            each link is checked at its link-frame point. If True, each link's
            AABB is converted to occupied voxels and all those voxel costs are
            checked.

    Returns:
        True if the configuration is under the voxel/contact cost threshold and
        has no blocking contact. False otherwise.
    """
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)


    if voxel_grid is not None:
        
        if ee_link_index is None or voxel_size is None or origin is None:
            raise ValueError("ee_link_index, voxel_size, and origin are required when voxel_grid is provided.")
        
        
        if not is_free_config(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, max_cost, whole_robot=whole_robot, use_all_occupied_voxels=use_all_occupied_voxels):
            return False
    
    print(f"The voxel grid cost is not provided, using mesh contact to determine validity")
    return not is_in_unsafe_contact(robot, cost_dict or {}, max_cost, ignored_body_ids=ignored_body_ids, ignored_robot_link_ids=ignored_robot_link_ids)
