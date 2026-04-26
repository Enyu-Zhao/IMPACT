import pybullet as p
import json
import os
import argparse
import numpy as np

from simulation.scene import Scene
from simulation.grasp_utils import (load_grasps, grasps_to_world,
                                     inject_grasps_into_voxel_grid,
                                     inject_grasps_into_2d_grid)
from planning.rrt import RRTVoxelGrid
from planning.rrt_star import RRTStarVoxelGrid
from planning.a_star import AStar
from planning.a_star_no_dir import PlainAStar
from planning.sample_pushing import analyze_push_safety, process_voxel_scene
from planning.bi_rrt import plan_bi_rrt
from planning.rrt_utils_config import is_valid_config


def find_path(scene_name, algo, cost_map, start_position, target_position,
              world_grasps=None, log_dir=None, filename=None):
    if algo == 'rrt_star':
        if world_grasps:
            cost_map.voxel_grid = inject_grasps_into_voxel_grid(cost_map.voxel_grid, world_grasps)
        rrt_star = RRTStarVoxelGrid(cost_map)
        path, cost = rrt_star.find_path(start_position)
    elif algo == "rrt":
        if world_grasps:
            cost_map.voxel_grid = inject_grasps_into_voxel_grid(cost_map.voxel_grid, world_grasps)
        rrt = RRTVoxelGrid(cost_map)
        path, cost = rrt.find_path(start_position)
    elif algo == "a_star":
        voxel_grid = cost_map.voxel_grid
        base_grid, updated_grid = analyze_push_safety(voxel_grid, visualize=False)
        if world_grasps:
            base_grid    = inject_grasps_into_2d_grid(base_grid,    world_grasps)
            updated_grid = inject_grasps_into_2d_grid(updated_grid, world_grasps)
        a_star = AStar(scene_name, base_grid, updated_grid, robot_init_position=start_position, target_position=target_position, log_dir=log_dir, filename=filename)
        path, cost = a_star.find_path()
    elif algo == "a_star_plain":
        voxel_grid = cost_map.voxel_grid
        base_grid = process_voxel_scene(voxel_grid, visualize=False)
        if world_grasps:
            base_grid = inject_grasps_into_2d_grid(base_grid, world_grasps)
        plain_a_star = PlainAStar(scene_name, base_grid, base_grid, robot_init_position=start_position, target_position=target_position, log_dir=log_dir, filename=filename)
        path, cost = plain_a_star.find_path()

    return path, cost


_ARM_DOF = 7
_IK_GOAL_TOLERANCE = 0.01

def find_path_bi_rrt(robot, ee_link_index, target_position, cost_dict, target_id=None):
    rng = np.random.default_rng(seed=0)
    q_init = [p.getJointState(robot, i)[0] for i in range(_ARM_DOF)]
    lower = np.array([p.getJointInfo(robot, i)[8] for i in range(_ARM_DOF)])
    upper = np.array([p.getJointInfo(robot, i)[9] for i in range(_ARM_DOF)])
    joint_ranges = upper - lower
    goals = []
    attempts = 0
    while len(goals) < 20 and attempts < 200:
        attempts += 1
        q_rand = rng.uniform(lower, upper)
        for i, pos in enumerate(q_rand):
            p.resetJointState(robot, i, pos)
        q_ik = np.array(p.calculateInverseKinematics(
            robot, ee_link_index, target_position,
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=joint_ranges.tolist(),
            restPoses=q_rand,
            maxNumIterations=300,
            residualThreshold=1e-4,
        ))[:_ARM_DOF]
        if np.any(q_ik < lower) or np.any(q_ik > upper):
            continue
        for i, pos in enumerate(q_ik):
            p.resetJointState(robot, i, pos)
        ee_pos = np.array(p.getLinkState(robot, ee_link_index)[4])
        if np.linalg.norm(ee_pos - np.array(target_position)) > _IK_GOAL_TOLERANCE:
            continue
        if not is_valid_config(robot, q_ik, cost_dict=cost_dict, ignored_body_ids=[target_id], ignored_robot_link_ids=[-1]):
            continue
        goals.append(q_ik)
    for i, pos in enumerate(q_init):
        p.resetJointState(robot, i, pos)
    print(f"[INFO] generated {len(goals)} valid IK goals from {attempts} attempts")
    path = plan_bi_rrt(robot, goals, cost_dict, target_id=target_id, seed=42)
    for i, pos in enumerate(q_init):
        p.resetJointState(robot, i, pos)
    if path is not None:
        path = [q.tolist() for q in path]
    return path


def main(args):
    log_dir = f"logs/{args.scene}/{args.algo}"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{args.scene}-{args.algo}"

    scene_file = f'assets/scenes/{args.scene}/scene.json'
    with open(scene_file, 'r') as json_file:
        scene_data = json.load(json_file)
    target_object_name = scene_data["target_object"]

    scene = Scene(scene_data=scene_data, name=args.scene)
    scene.start_physics_client(gui=False)
    scene.load_all_models()

    log_scene_file = f'{log_dir}/{filename}_scene.json'
    cost_map = scene.generate_cost_map(log_scene_file, target_object_name)
    cost_map.visualize_voxel_grid()

    robot, ee_link_index = scene.load_robot()

    if args.algo == 'bi_rrt':
        path = find_path_bi_rrt(robot, ee_link_index, scene.target_position, scene.cost_dict, target_id=scene.target_id)
        cost = 0
    else:
        initial_end_effector_position = p.getLinkState(robot, ee_link_index)[4]
        path, cost = find_path(args.scene, args.algo, cost_map, initial_end_effector_position, scene.target_position, log_dir, filename)

    if path:
        path_file = f'{log_dir}/{filename}_path.json'
        with open(path_file, 'w') as json_file:
            json.dump({"path": path, "cost": cost}, json_file)
        print(f"Path and cost saved to {path_file}")

    else:
        print("No path found.")

    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="scene01")
    parser.add_argument('--algo', type=str, default="a_star")
    args = parser.parse_args()

    main(args)
