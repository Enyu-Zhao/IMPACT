import pybullet as p
import numpy as np

from planning.rrt_utils import world_to_grid

# 1. Voxel Method
# 1.1 function for getting the cost of a configuration based on the end-effector position in the voxel grid
def ee_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin):
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)
    ee_pos = np.array(p.getLinkState(robot, ee_link_index)[4])
    x, y, z = world_to_grid(ee_pos, voxel_size, origin)
    x_dim, y_dim, z_dim = voxel_grid.shape
    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
        return voxel_grid[x, y, z]
    return float('inf')

# 1.2 functions for getting the cost of a configuration based on the maximum voxel cost at the robot's occupied voxels
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

#1.2.1 helper function for robot_max_voxel_cost that gets the costs of all occupied voxels by the robot
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



# 1.3 function for checking if a configuration is valid based on voxel cost
def config_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, whole_robot=False, use_all_occupied_voxels=False):
    if whole_robot:
        return robot_max_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, use_all_occupied_voxels=use_all_occupied_voxels)
    return ee_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin)


def is_free_config(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, max_cost, whole_robot=False, use_all_occupied_voxels=False):
    return config_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, whole_robot=whole_robot, use_all_occupied_voxels=use_all_occupied_voxels) <= max_cost



# 2. Mesh method (ideally shouldn't be used.Only use as a fallback)
# 2.1 function for getting the cost of a configuration based on the maximum object cost currently contacted by the robot
def contact_config_cost(robot, cost_dict=None, ignored_body_ids=None, ignored_robot_link_ids=None):
    """Return the maximum object cost currently contacted by the robot."""
    cost_dict = {} if cost_dict is None else cost_dict
    ignored_body_ids = set() if ignored_body_ids is None else set(ignored_body_ids)
    ignored_robot_link_ids = set() if ignored_robot_link_ids is None else set(ignored_robot_link_ids)

    p.performCollisionDetection()
    max_cost = 0
    for contact in p.getContactPoints(bodyA=robot):
        body_b = contact[2]
        robot_link = contact[3]
        if robot_link in ignored_robot_link_ids or body_b == robot or body_b in ignored_body_ids:
            continue
        cost = cost_dict.get(body_b, float('inf'))
        if cost == -1:
            continue
        max_cost = max(max_cost, cost)
    return max_cost


def is_in_unsafe_contact(robot, cost_dict, max_cost, ignored_body_ids=None, ignored_robot_link_ids=None):
    return contact_config_cost(robot, cost_dict, ignored_body_ids=ignored_body_ids, ignored_robot_link_ids=ignored_robot_link_ids) > max_cost

# 3. Overall config cost function API to call. 
def config_cost(
    robot,
    joint_positions,
    cost_dict=None,
    ignored_body_ids=None,
    ignored_robot_link_ids=None,
    ee_link_index=None,
    voxel_grid=None,
    voxel_size=None,
    origin=None,
    whole_robot=False,
    use_all_occupied_voxels=False):
    """Return the scalar cost of a robot joint configuration.

    The returned cost is the maximum of:
    1. Contact cost from bodies currently touched by the robot.
    2. Voxel cost, enabled when voxel_grid is provided.

    Args:
        robot: PyBullet body id for the robot.
        joint_positions: Joint values for the robot arm configuration.
        cost_dict: Optional mapping from PyBullet body id to object cost.
            Missing body ids are treated as infinite cost, so contact with them
            is blocking. A cost of -1 is treated as the target and allowed.
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
        A numeric configuration cost. A caller can compare this value against a
        threshold to decide whether the configuration is valid.
    """
    
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot, i, pos)

    cost = contact_config_cost(robot, cost_dict, ignored_body_ids=ignored_body_ids, ignored_robot_link_ids=ignored_robot_link_ids)
    if voxel_grid is None:
        return cost
    if ee_link_index is None or voxel_size is None or origin is None:
        raise ValueError("ee_link_index, voxel_size, and origin are required when voxel_grid is provided.")

    voxel_cost = config_voxel_cost(robot, joint_positions, ee_link_index, voxel_grid, voxel_size, origin, whole_robot=whole_robot, use_all_occupied_voxels=use_all_occupied_voxels)
    return max(cost, voxel_cost)



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
    
    """Return whether a robot joint configuration is valid based on a cost threshold. A wrapper for config_cost for convenience."""
    
    cost = config_cost(
        robot,
        joint_positions,
        cost_dict=cost_dict,
        ignored_body_ids=ignored_body_ids,
        ignored_robot_link_ids=ignored_robot_link_ids,
        ee_link_index=ee_link_index,
        voxel_grid=voxel_grid,
        voxel_size=voxel_size,
        origin=origin,
        whole_robot=whole_robot,
        use_all_occupied_voxels=use_all_occupied_voxels,
    )
    return cost <= max_cost


# 4. Functions for executing paths with smooth interpolation in joint space or Cartesian space. Only used for testing. Might not be useful.
def generate_valid_ik_goals(
    robot,
    ee_link_index,
    target_position,
    cost_dict=None,
    target_id=None,
    arm_dof=7,
    num_goals=20,
    max_attempts=200,
    seed=0,
    ik_goal_tolerance=0.01,
    max_cost=9,
    voxel_grid=None,
    voxel_size=None,
    origin=None,
    whole_robot=False,
    use_all_occupied_voxels=False,
):
    """Generate collision-valid IK goal configurations for a Cartesian target."""
    rng = np.random.default_rng(seed=seed)
    q_init = [p.getJointState(robot, i)[0] for i in range(arm_dof)]
    lower = np.array([p.getJointInfo(robot, i)[8] for i in range(arm_dof)])
    upper = np.array([p.getJointInfo(robot, i)[9] for i in range(arm_dof)])
    joint_ranges = upper - lower

    goals = []
    attempts = 0
    while len(goals) < num_goals and attempts < max_attempts:
        attempts += 1
        q_rand = rng.uniform(lower, upper)
        for i, pos in enumerate(q_rand):
            p.resetJointState(robot, i, pos)

        q_ik = np.array(p.calculateInverseKinematics(
            robot,
            ee_link_index,
            target_position,
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=joint_ranges.tolist(),
            restPoses=q_rand,
            maxNumIterations=300,
            residualThreshold=1e-4,
        ))[:arm_dof]

        if np.any(q_ik < lower) or np.any(q_ik > upper):
            continue

        for i, pos in enumerate(q_ik):
            p.resetJointState(robot, i, pos)
        ee_pos = np.array(p.getLinkState(robot, ee_link_index)[4])
        if np.linalg.norm(ee_pos - np.array(target_position)) > ik_goal_tolerance:
            continue

        if not is_valid_config(
            robot,
            q_ik,
            cost_dict=cost_dict,
            max_cost=max_cost,
            ignored_body_ids=[] if target_id is None else [target_id],
            ignored_robot_link_ids=[-1],
            ee_link_index=ee_link_index,
            voxel_grid=voxel_grid,
            voxel_size=voxel_size,
            origin=origin,
            whole_robot=whole_robot,
            use_all_occupied_voxels=use_all_occupied_voxels,
        ):
            continue

        goals.append(q_ik)

    for i, pos in enumerate(q_init):
        p.resetJointState(robot, i, pos)

    return goals, attempts

