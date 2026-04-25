"""Example of how to run the bi-directional RRT planner to get a plan."""

from simulation.scene import Scene
import json
from planning.bi_rrt import plan_bi_rrt, random_arm_config, set_arm_config
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

    robot, ee_link_index = scene.load_robot()

    # TODO: remove the rng seeds? Currently have them fixed for debugging
    goals = [random_arm_config(robot, seed=i) for i in range(3)]
    plan = plan_bi_rrt(robot, goals, seed=42)

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
