import json
import os
import numpy as np
from scipy.spatial.transform import Rotation


'''USAGE:
from simulation.grasp_utils import load_grasps, grasps_to_world, grasps_to_joint_configs

grasp_data  = load_grasps(target_object_name, scale)
world_grasps = grasps_to_world(grasp_data, position, orientation_euler)

# needs an active PyBullet client with the robot already loaded
joint_configs = grasps_to_joint_configs(world_grasps, robot, ee_link_index)
# → 100 dicts, each with grasp_id, solution_idx (0/1), score, joint_config [q0..q6]
'''



GRASP_DIR = "assets/grasps"
# A* grid parameters (must match convert_astar_path_to_world)
_ASTAR_BASE_X  = 0.6
_ASTAR_BASE_Y  = 0.0
_ASTAR_VSIZE   = 0.01
_ASTAR_ROW0    = 97
_ASTAR_COL0    = 50
# 3-D voxel grid parameters (must match CostMap._generate_voxels)
_VOX_ORIGIN    = np.array([-0.5, -0.5, 0.0])
_VOX_SIZE      = np.array([0.01, 0.01, 0.01])


def load_grasps(object_name, scale, grasp_dir=GRASP_DIR):
    """Load grasp JSON for object_name at scale; falls back to nearest scale."""
    # try integer scale too (e.g. scale=2.0 → "scale_2")
    candidates = [f"{object_name}_scale_{scale}.json"]
    if float(scale) == int(float(scale)):
        candidates.append(f"{object_name}_scale_{int(float(scale))}.json")

    for fname in candidates:
        path = os.path.join(grasp_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    # fuzzy: pick closest scale among files for this object
    prefix = f"{object_name}_scale_"
    try:
        matches = [fn for fn in os.listdir(grasp_dir)
                   if fn.startswith(prefix) and fn.endswith(".json")]
    except FileNotFoundError:
        return None
    if not matches:
        return None

    best = min(matches, key=lambda fn: abs(float(fn[len(prefix):-5]) - float(scale)))
    print(f"[grasps] no exact match for {object_name} scale={scale}, using {best}")
    with open(os.path.join(grasp_dir, best)) as f:
        return json.load(f)


def object_transform(position, orientation_euler):
    """4x4 world transform from position and XYZ euler angles (radians)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', orientation_euler).as_matrix()
    T[:3, 3] = position
    return T


def grasps_to_world(grasp_data, object_position, object_orientation_euler):
    """Return list of dicts {grasp_id, score, transform (4x4 np array)} in world frame."""
    T_obj = object_transform(object_position, object_orientation_euler)
    out = []
    for g in grasp_data["grasps"]:
        T_world = T_obj @ np.array(g["transform"])
        out.append({"grasp_id": g["grasp_id"], "score": g["score"], "transform": T_world})
    return out


def grasp_position(T):
    return np.array(T)[:3, 3]


def grasp_approach(T):
    """Z-axis of grasp frame = approach direction (GraspGen convention)."""
    return np.array(T)[:3, 2]


def grasp_yaw_deg(T):
    return float(np.degrees(Rotation.from_matrix(np.array(T)[:3, :3]).as_euler('xyz')[2]))


def best_grasp(world_grasps, ee_position):
    """Grasp closest to ee_position, tie-broken by score."""
    ee = np.array(ee_position)
    return min(world_grasps,
               key=lambda g: np.linalg.norm(grasp_position(g["transform"]) - ee)
                             / (g["score"] + 1e-6))


def inject_grasps_into_voxel_grid(voxel_grid, world_grasps, standoff=0.05):
    """Mark grasp approach positions as -1 (TARGET) in the 3-D voxel grid."""
    grid = voxel_grid.copy()
    dims = np.array(grid.shape)
    for g in world_grasps:
        T = g["transform"]
        pos = grasp_position(T) - grasp_approach(T) * standoff
        idx = np.round((pos - _VOX_ORIGIN) / _VOX_SIZE).astype(int)
        if np.all(idx >= 0) and np.all(idx < dims):
            grid[idx[0], idx[1], idx[2]] = -1
    return grid


# Two structurally distinct rest poses for IK seeding (elbow-up vs elbow-down)
_REST_POSES = [
    [ 0.05, -1.15, -0.13, -3.06, -0.05,  2.05, -2.35],
    [-0.05, -0.80,  0.10, -2.50,  0.10,  1.50, -0.80],
]
_IK_LOWER  = [-2.8973,-1.7628,-2.8973,-3.0718,-2.8973, 0.0,   -2.8973]
_IK_UPPER  = [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]
_IK_RANGES = [ 5.8,    4.0,    5.8,    6.0,    5.8,    6.5,    5.8   ]


def grasps_to_joint_configs(world_grasps, robot, ee_link_index, rest_poses=None):
    """
    Compute 2 IK solutions per grasp → 100 joint configs total (2 × 50).

    Each entry in the returned list:
        {
            "grasp_id":     int,
            "solution_idx": 0 or 1,
            "score":        float,
            "joint_config": [q0..q6]   # 7-dim, radians
        }

    rest_poses: list of two 7-dim seed configs; defaults to elbow-up / elbow-down pair.
    """
    import pybullet as p

    if rest_poses is None:
        rest_poses = _REST_POSES

    configs = []
    for g in world_grasps:
        T    = np.array(g["transform"])
        pos  = T[:3, 3].tolist()
        quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()  # [x,y,z,w]

        for idx, rest in enumerate(rest_poses):
            q = p.calculateInverseKinematics(
                robot, ee_link_index, pos, quat,
                lowerLimits=_IK_LOWER,
                upperLimits=_IK_UPPER,
                jointRanges=_IK_RANGES,
                restPoses=rest,
                maxNumIterations=300,
                residualThreshold=1e-4,
            )
            configs.append({
                "grasp_id":     g["grasp_id"],
                "solution_idx": idx,
                "score":        g["score"],
                "joint_config": list(q[:7]),
            })

    return configs  # length = len(world_grasps) * len(rest_poses)


def inject_grasps_into_2d_grid(grid_2d, world_grasps, standoff=0.05):
    """Mark grasp approach positions as -1 (TARGET) in the 2-D A* base map."""
    grid = grid_2d.copy()
    h, w = grid.shape
    for g in world_grasps:
        T = g["transform"]
        pos = grasp_position(T) - grasp_approach(T) * standoff
        x, y = pos[0], pos[1]
        row = int(round(_ASTAR_ROW0 - (x - (_ASTAR_BASE_X - 0.2)) / _ASTAR_VSIZE))
        col = int(round(_ASTAR_COL0 + y / _ASTAR_VSIZE))
        if 0 <= row < h and 0 <= col < w:
            grid[row, col] = -1
    return grid
