import pybullet as p
import json
import argparse
import os
import time
import numpy as np

from planning.rrt_utils import draw_path, interpolate_path, smooth_path
from simulation.scene import Scene
from simulation.evaluation import evaluate_path, ik_pose_xyz, ik_pose_xyz_yaw, shortest_wrap
from simulation.grasp_utils import load_grasps, grasps_to_world, best_grasp, grasp_position, grasp_approach, grasp_yaw_deg

FINGER_OPEN   = 0.04
FINGER_CLOSED = 0.0
FINGER_FORCE  = 150


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


def _set_arm(robot, q, steps, finger_pos=None, realtime=False):
    for j in range(7):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                targetPosition=float(q[j]), force=500)
    for _ in range(steps):
        if finger_pos is not None:
            p.setJointMotorControl2(robot, 9,  p.POSITION_CONTROL,
                                    targetPosition=finger_pos, force=FINGER_FORCE)
            p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL,
                                    targetPosition=finger_pos, force=FINGER_FORCE)
        p.stepSimulation()
        if realtime:
            time.sleep(1 / 240)


def execute_grasp_and_lift(robot, ee_link_index, grasp_transform, rest_pose,
                           target_id=None, standoff=0.08, lift_height=0.2, move_steps=300):
    T        = np.array(grasp_transform)
    gpos     = grasp_position(T)
    approach = grasp_approach(T)
    yaw      = grasp_yaw_deg(T)

    pregrasp = (gpos - approach * standoff).tolist()
    grasp    = gpos.tolist()
    lift     = (gpos + np.array([0, 0, lift_height])).tolist()

    # set target to a graspable mass (load_all_models sets everything to 100 kg)
    if target_id is not None:
        p.changeDynamics(target_id, -1, mass=0.3)

    # open fingers
    for _ in range(80):
        p.setJointMotorControl2(robot, 9,  p.POSITION_CONTROL,
                                targetPosition=FINGER_OPEN, force=FINGER_FORCE)
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL,
                                targetPosition=FINGER_OPEN, force=FINGER_FORCE)
        p.stepSimulation()
        time.sleep(1 / 240)

    # move to pre-grasp
    q = ik_pose_xyz_yaw(robot, ee_link_index, pregrasp, yaw, rest_pose)
    _set_arm(robot, q, move_steps, finger_pos=FINGER_OPEN, realtime=True)

    # move to grasp pose
    q = ik_pose_xyz_yaw(robot, ee_link_index, grasp, yaw, rest_pose)
    _set_arm(robot, q, move_steps, finger_pos=FINGER_OPEN, realtime=True)

    # close fingers slowly
    for _ in range(250):
        p.setJointMotorControl2(robot, 9,  p.POSITION_CONTROL,
                                targetPosition=FINGER_CLOSED, force=FINGER_FORCE)
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL,
                                targetPosition=FINGER_CLOSED, force=FINGER_FORCE)
        p.stepSimulation()
        time.sleep(1 / 240)

    # attach object to gripper with a fixed constraint so it lifts reliably
    constraint_id = None
    if target_id is not None:
        constraint_id = p.createConstraint(
            robot, ee_link_index,
            target_id, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        )

    # lift slowly while keeping fingers closed
    q = ik_pose_xyz_yaw(robot, ee_link_index, lift, yaw, rest_pose)
    _set_arm(robot, q, move_steps + 100, finger_pos=FINGER_CLOSED, realtime=True)

    # hold at top to show success
    for _ in range(200):
        p.setJointMotorControl2(robot, 9,  p.POSITION_CONTROL,
                                targetPosition=FINGER_CLOSED, force=FINGER_FORCE)
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL,
                                targetPosition=FINGER_CLOSED, force=FINGER_FORCE)
        p.stepSimulation()
        time.sleep(1 / 240)

    if constraint_id is not None:
        p.removeConstraint(constraint_id)


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

    # load grasps for the target object
    obj_info = scene_data["object_models"][target_object_name]
    grasp_data = load_grasps(target_object_name, obj_info["scale"])
    world_grasps = None
    if grasp_data:
        world_grasps = grasps_to_world(
            grasp_data,
            obj_info["position"],
            obj_info.get("orientation", [0, 0, 0]),
        )

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

    try:
        if args.algo == 'a_star':
            execute_smooth_xyz_yaw(robot, ee_link_index, path, scene.initial_joint_positions, samples=40)
        else:
            execute_smooth_xyz(robot, ee_link_index, path, scene.initial_joint_positions, samples=40)

        if world_grasps:
            ee_pos = p.getLinkState(robot, ee_link_index)[4]
            chosen = best_grasp(world_grasps, ee_pos)
            print(f"[grasps] executing grasp {chosen['grasp_id']} (score={chosen['score']:.3f})")
            execute_grasp_and_lift(robot, ee_link_index, chosen["transform"],
                                   scene.initial_joint_positions,
                                   target_id=scene.target_id)
    except Exception as e:
        import traceback
        print(f"[ERROR] during execution: {e}")
        traceback.print_exc()
    finally:
        p.stopStateLogging(log_id)
        p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="scene01")
    parser.add_argument('--algo', type=str, default="a_star")
    args = parser.parse_args()

    main(args)
