"""Example of how to run the bi-directional RRT planner to get a plan."""
import sys
from simulation.scene import Scene
import json
from planning.bi_rrt import plan_bi_rrt, set_arm_config
from planning.rrt_utils_config import is_valid_config
import numpy as np
import pybullet as p
import time

_SCENE = "scene01"
_GUI = True

if __name__ == "__main__":
    scene_file = f"assets/scenes/{_SCENE}/scene.json"
    with open(scene_file, 'r') as json_file:
        scene_data = json.load(json_file)

    scene = Scene(scene_data=scene_data, name=_SCENE)
    scene.start_physics_client(gui=_GUI)
    scene.load_all_models()

    target_object_name = scene_data["target_object"]
    cost_map = scene.generate_cost_map(scene_file=scene_file, target_object_name=target_object_name)
    cost_dict = scene.cost_dict

    robot, ee_link_index = scene.load_robot()

    _PANDA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.0,    -2.8973]
    _PANDA_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
    _PANDA_RANGE = [upper - lower for lower, upper in zip(_PANDA_LOWER, _PANDA_UPPER)]
    _IK_GOAL_TOLERANCE = 0.01

    rng = np.random.default_rng(seed=0)
    q_init = [p.getJointState(robot, i)[0] for i in range(7)]
    lower, upper = np.array(_PANDA_LOWER), np.array(_PANDA_UPPER)
    goals = []
    attempts = 0
    while len(goals) < 20 and attempts < 200:
        attempts += 1
        q_rand = rng.uniform(_PANDA_LOWER, _PANDA_UPPER)
        for i, pos in enumerate(q_rand):
            p.resetJointState(robot, i, pos)
        q_ik = list(p.calculateInverseKinematics(
            robot, ee_link_index, scene.target_position,
            lowerLimits=_PANDA_LOWER,
            upperLimits=_PANDA_UPPER,
            jointRanges=_PANDA_RANGE,
            restPoses=q_rand,
            maxNumIterations=300,
            residualThreshold=1e-4,
        ))[:7]
        q_ik = np.array(q_ik)
        if np.any(q_ik < lower) or np.any(q_ik > upper):
            continue
        for i, pos in enumerate(q_ik):
            p.resetJointState(robot, i, pos)
        ee_pos = np.array(p.getLinkState(robot, ee_link_index)[4])
        if np.linalg.norm(ee_pos - np.array(scene.target_position)) > _IK_GOAL_TOLERANCE:
            continue
        if not is_valid_config(robot, q_ik, cost_dict=cost_dict, ignored_body_ids=[scene.target_id], ignored_robot_link_ids=[-1]):
            continue
        goals.append(np.array(q_ik))
    for i, pos in enumerate(q_init):
        p.resetJointState(robot, i, pos)
    print(f"[INFO] generated {len(goals)} valid IK goals from {attempts} attempts")

    plan = plan_bi_rrt(robot, goals, cost_dict=cost_dict, target_id=scene.target_id, seed=42)

    if plan is None:
        print("planning failed")
        sys.exit(1)
    # Emulate a trajectory from the path by performing kinematic updates at a fixed rate
    playback_frequency = 30 # hz
    timestep = 1 / playback_frequency # seconds
    # pause briefly at the initial configuration
    # TODO: figure out why this doesn't seemt to give the proper visualization update in the GUI (rest of the trajectory visualization seems good)
    set_arm_config(robot, plan[0])
    time.sleep(1)
    for q in plan[1:]:
        start_time = time.time()
        set_arm_config(robot, q)
        time_until_next_step = timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    # pause briefly at the end of the emulated trajectory
    time.sleep(1)
