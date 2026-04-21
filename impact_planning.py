import pybullet as p
import json
import os
import argparse

from simulation.scene import Scene
from simulation.grasp_utils import (load_grasps, grasps_to_world,
                                     inject_grasps_into_voxel_grid,
                                     inject_grasps_into_2d_grid)
from planning.rrt import RRTVoxelGrid
from planning.rrt_star import RRTStarVoxelGrid
from planning.a_star import AStar
from planning.a_star_no_dir import PlainAStar
from planning.sample_pushing import analyze_push_safety, process_voxel_scene


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
    initial_end_effector_position = p.getLinkState(robot, ee_link_index)[4]

    # Load grasps for the target object and transform to world frame
    obj_info = scene_data["object_models"][target_object_name]
    grasp_data = load_grasps(target_object_name, obj_info["scale"])
    world_grasps = None
    if grasp_data:
        world_grasps = grasps_to_world(
            grasp_data,
            obj_info["position"],
            obj_info.get("orientation", [0, 0, 0]),
        )
        print(f"[grasps] loaded {len(world_grasps)} grasps for {target_object_name}")
    else:
        print(f"[grasps] no grasp file found for {target_object_name}, using object voxels as goal")

    path, cost = find_path(args.scene, args.algo, cost_map, initial_end_effector_position,
                           scene.target_position, world_grasps=world_grasps,
                           log_dir=log_dir, filename=filename)

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
