import pybullet as p
import json
import argparse
import os
import numpy as np

from planning.rrt_utils import draw_path, interpolate_path, smooth_path
from simulation.scene import Scene
from simulation.evaluation import evaluate_path, ik_pose_xyz, ik_pose_xyz_yaw, shortest_wrap


def execute_smooth_xyz(robot, ee_link_index, xyz_path, rest_pose, samples=40, settle_steps=100):
    p.setRealTimeSimulation(0)
    q_key = np.vstack([ik_pose_xyz(robot, ee_link_index, pt[:3], rest_pose) for pt in xyz_path])
    trajectory = [q_key[0]]
    for q0, q1 in zip(q_key[:-1], q_key[1:]):
        dq = shortest_wrap(q0, q1)
        seg = q0 + dq * np.linspace(0, 1, samples, dtype=float)[:, None]
        trajectory.extend(seg[1:])
    for q in trajectory:
        for j in range(7):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=float(q[j]), force=500)
        p.stepSimulation()

    for _ in range(settle_steps):
        p.stepSimulation()


def execute_smooth_xyz_yaw(robot, ee_link_index, xyz_yaw_path, rest_pose, samples=40, settle_steps=100):
    p.setRealTimeSimulation(0)
    q_key = np.vstack([ik_pose_xyz_yaw(robot, ee_link_index, point[:3], point[3], rest_pose)
                       for point in xyz_yaw_path])

    trajectory = [q_key[0]]
    for q0, q1 in zip(q_key[:-1], q_key[1:]):
        dq = shortest_wrap(q0, q1)
        seg = q0 + dq * np.linspace(0, 1, samples, dtype=float)[:, None]
        trajectory.extend(seg[1:])  

    for q in trajectory:
        for j in range(7):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=float(q[j]), force=500)
        p.stepSimulation()

    for _ in range(settle_steps):
        p.stepSimulation()


def main(args):
    log_dir = f"logs/{args.scene}/{args.algo}"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{args.scene}-{args.algo}"

    scene_file = f"{log_dir}/{filename}_scene.json"
    if not os.path.exists(scene_file):
        scene_file = f"assets/scenes/{args.scene}/scene.json"

    with open(scene_file, 'r') as json_file:
        scene_data = json.load(json_file)
    target_object_name = scene_data["target_object"]

    try:
        with open(f'{log_dir}/{filename}_path.json', 'r') as path_file:
            path = json.load(path_file)["path"]
    except FileNotFoundError:
        print("No path.")
        return

    scene = Scene(scene_data=scene_data, name=args.scene)
    scene.start_physics_client(gui=True)
    scene.load_all_models()

    scene.load_cost_map(target_object_name)

    robot, ee_link_index = scene.load_robot()
    obstacles = scene.get_obstacles()
  
    steps = 5
    path = smooth_path(path, window_size=5)
    path = interpolate_path(path, steps)  

    evaluation_data = evaluate_path(path, robot, obstacles, ee_link_index, scene)
    with open(f"{log_dir}/{filename}_metrics.json", "w") as f:
        json.dump(evaluation_data, f, indent=4) 

    draw_path(path)
    p.resetDebugVisualizerCamera(0.6, 270, -30, [0.2, 0, 0.3])
    for _ in range(30):
        p.stepSimulation()

    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"{log_dir}/{filename}_video.mp4")
    if args.algo == 'a_star':
        execute_smooth_xyz_yaw(robot, ee_link_index, path, scene.initial_joint_positions, samples=40)
    else:
        execute_smooth_xyz(robot, ee_link_index, path, scene.initial_joint_positions, samples=40)
    p.stopStateLogging(log_id)

    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="scene01")
    parser.add_argument('--algo', type=str, default="a_star")
    args = parser.parse_args()

    main(args)