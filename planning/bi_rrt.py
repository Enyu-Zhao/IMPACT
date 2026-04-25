from planning.tree import Node, Tree
import pybullet as p
import numpy as np

_ARM_DOF = 7

def _get_joint_limits(robot) -> tuple[list[float], list[float]]:
    lower = []
    upper = []
    for i in range(_ARM_DOF):
        info = p.getJointInfo(robot, i)
        lower.append(info[8])
        upper.append(info[9])
    return lower, upper

def set_arm_config(robot, arm_config) -> None:
    for i, pos in zip(range(_ARM_DOF), arm_config):
        p.resetJointState(robot, i, pos)

def _contact_points(robot):
    """returns a list of contact points for the robot. Empty list means robot is collision free."""
    p.performCollisionDetection()
    return p.getContactPoints(robot)

def _get_current_arm_config(robot) -> np.ndarray:
    arm_joint_states = p.getJointStates(robot, list(range(_ARM_DOF)))
    return np.array([state[0] for state in arm_joint_states])

def random_arm_config(robot, seed: int | None = None, collision_free = True) -> np.ndarray:
    lower, upper = _get_joint_limits(robot)
    rng = np.random.default_rng(seed=seed)

    q_init = _get_current_arm_config(robot)
    while True:
        q_rand = rng.uniform(lower, upper)
        if not collision_free:
            break
        else:
            set_arm_config(robot, q_rand)
            if not _contact_points(robot):
                break
    # reset robot to its original position once a valid random config has been found
    set_arm_config(robot, q_init)
    return q_rand

def _step(start: np.ndarray, target: np.ndarray, max_step_dist: float) -> np.ndarray:
    """Step a vector towards a target.

    Args:
        start: The start vector.
        target: The target.
        max_step_dist: Maximum amount to step towards `target`.

    Return:
        A vector that has taken a step towards `target` from `start`.
    """
    if max_step_dist <= 0.0:
        raise ValueError("`max_step_dist` must be > 0.0")
    if np.array_equal(start, target):
        return start.copy()
    direction = target - start
    magnitude = np.linalg.norm(direction)
    unit_vec = direction / magnitude
    return start + (unit_vec * min(max_step_dist, magnitude))

def _connect(q_target, tree, epsilon, robot) -> np.ndarray:
    closest_node = tree.nearest_neighbor(q_target)
    q = closest_node.q
    while True:
        if np.array_equal(q, q_target):
            # reached the target
            return q
        q_next = _step(q, q_target, epsilon)
        set_arm_config(robot, q_next)
        if _contact_points(robot):
            # in collision, so return the previous (valid) config
            return q
        closest_node = Node(q_next, closest_node)
        tree.add_node(closest_node)
        q = q_next

def _combine_paths(
    start_tree: Tree,
    start_tree_node: Node,
    goal_tree: Tree,
    goal_tree_node: Node,
) -> list[np.ndarray]:
    """Combine paths from a start and goal tree.

    Args:
        start_tree: The tree whose root is the start of the combined path.
        start_tree_node: The node in `start_tree` that marks the end of the
            path that begins at the root of `start_tree`.
        goal_tree: The tree whose root is the end of the combined path.
        goal_tree_node: The node in `goal_tree` that marks the beginning of
            the path that ends at the root of `goal_tree`.

    Returns:
        A path that starts at the root of `start_tree` and ends at `goal_tree`,
        with a connecting edge between `start_tree_node` and `goal_tree_node`.
    """
    # The path generated from start_tree ends at q_init, but we want it to
    # start at q_init. So we must reverse it.
    path_start = [n.q for n in start_tree.get_path(start_tree_node)]
    path_start.reverse()
    # The path generated from goal_tree ends at q_goal, which is what we want.
    path_end = [n.q for n in goal_tree.get_path(goal_tree_node)]
    # The last value in path_start might be the same as the first value in
    # path_end. If this is the case, remove the duplicate value.
    if np.array_equal(path_start[-1], path_end[0]):
        path_start.pop()
    return path_start + path_end

def plan_bi_rrt(robot, goal_arm_configs, seed: int | None = None, epsilon = 0.05, goal_biasing_probability = 0.05, max_iters=5000) -> list[np.ndarray]:
    q_init = _get_current_arm_config(robot)

    for q_goal in goal_arm_configs:
        set_arm_config(robot, q_goal)
        assert not _contact_points(robot), f"goal config {q_goal} is in collision"

    joint_limits = _get_joint_limits(robot)

    start_tree = Tree(Node(q_init))
    # To support multiple goals, the root of the goal tree is a sink node and all
    # goal configurations are children of this sink node. Setting the sink node to a
    # configuration with all values being infinity emulates exclusion from nearest
    # neighbor search in the tree, since the distance to this node is infinity.
    sink_node = Node(np.ones_like(q_init) * np.inf)
    goal_nodes = [Node(q, sink_node) for q in goal_arm_configs]
    goal_tree = Tree(sink_node)
    for n in goal_nodes:
        goal_tree.add_node(n)

    rng = np.random.default_rng(seed=seed)
    tree_a, tree_b = start_tree, goal_tree
    swapped = False

    for _ in range(max_iters):
        if rng.random() <= goal_biasing_probability:
            if swapped:
                q_rand = q_init
            else:
                # randomly pick a goal
                random_goal_idx = rng.integers(0, len(goal_nodes))
                q_rand = goal_nodes[random_goal_idx].q
        else:
            # sample a random config
            q_rand = rng.uniform(joint_limits[0], joint_limits[1])

        q_reached_a = _connect(q_rand, tree_a, epsilon, robot)
        q_reached_b = _connect(q_reached_a, tree_b, epsilon, robot)
        if np.array_equal(q_reached_a, q_reached_b):
            waypoints = _combine_paths(
                start_tree,
                start_tree.nearest_neighbor(q_reached_a),
                goal_tree,
                goal_tree.nearest_neighbor(q_reached_a),
            )
            # Ignore the last value which corresponds to the goal tree's sink node.
            return waypoints[:-1]

        # swap trees
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped
