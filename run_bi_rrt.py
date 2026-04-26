"""Example of how to run the bi-directional RRT planner to get a plan."""
import sys
from simulation.scene import Scene
import json
from planning.bi_rrt import plan_bi_rrt, set_arm_config
from planning.rrt_utils_config import generate_valid_ik_goals
import pybullet as p
import time

_SCENE = "scene01"
_GUI = True
_MAX_COST = 9

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

    q_init = [p.getJointState(robot, i)[0] for i in range(7)]
    goals, attempts = generate_valid_ik_goals(robot, ee_link_index, scene.target_position, cost_dict=cost_dict, target_id=scene.target_id, max_cost=_MAX_COST, voxel_grid=cost_map.voxel_grid, voxel_size=cost_map.voxel_size, origin=cost_map.origin, whole_robot=True, use_all_occupied_voxels=True)
    for i, pos in enumerate(q_init):
        p.resetJointState(robot, i, pos)
    print(f"[INFO] generated {len(goals)} valid IK goals from {attempts} attempts")

    plan = plan_bi_rrt(robot, goals, cost_dict=cost_dict, target_id=scene.target_id, seed=42, max_cost=_MAX_COST, cost_map=cost_map, ee_link_index=ee_link_index, whole_robot=True, use_all_occupied_voxels=True)

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
