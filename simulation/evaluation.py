import pybullet as p
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from planning.rrt_utils import get_initial_robot_state


MAX_DURATION = 20


def compute_oriented_gripper(yaw_deg):
    return (R.from_euler("z", np.radians(yaw_deg)) *
            R.from_euler("x", np.pi)).as_quat()


def ik_pose_xyz(robot, ee_link_index, xyz, rest):
    q = p.calculateInverseKinematics(robot, ee_link_index, xyz,
        lowerLimits=[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973, 0.0,-2.8973],
        upperLimits=[ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973],
        restPoses=rest, maxNumIterations=300, residualThreshold=1e-4)
    return np.array(q[:7])


def ik_pose_xyz_yaw(robot, ee_link_index, xyz, yaw_deg, rest):
    quat = compute_oriented_gripper(yaw_deg)
    q = p.calculateInverseKinematics(
        robot, ee_link_index, xyz, targetOrientation=quat,
        lowerLimits=[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973, 0.0,-2.8973],
        upperLimits=[ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973],
        jointRanges=[5.8,4.0,5.8,6.0,5.8,6.5,5.8],
        restPoses=rest, maxNumIterations=300, residualThreshold=1e-4)
    return np.asarray(q[:7])


def shortest_wrap(q0, q1):
    return (q1 - q0 + np.pi) % (2*np.pi) - np.pi


def evaluate_path(path, robot, obstacles, ee_link_index, scene, contact_index=[9, 10], samples=40):
    metric_data = {}
    target_reached = False
    path_cost = 0
    total_collided_obstacles = set()
    total_collision_time = 0
    reach_target_time = 0
    displacements = {}
    min_distance = {} 
    for obj in scene.scene_objects:
        min_distance[obj.name] = float('inf')

    control_force = 500
    step_time = p.getPhysicsEngineParameters()['fixedTimeStep']
    initial_state = get_initial_robot_state(robot)
    start_time = time.time()

    rest_pose = scene.initial_joint_positions

    p.setRealTimeSimulation(0)
    if len(path[0]) == 4:  # If path includes yaw
        q_key = np.vstack([ik_pose_xyz_yaw(robot, ee_link_index, point[:3], point[3], rest_pose) for point in path])
    else: 
        q_key = np.vstack([ik_pose_xyz(robot, ee_link_index, point[:3], rest_pose) for point in path])

    trajectory = [q_key[0]]
    for q0, q1 in zip(q_key[:-1], q_key[1:]):
        dq = shortest_wrap(q0, q1)
        seg = q0 + dq * np.linspace(0, 1, samples, dtype=float)[:, None]
        trajectory.extend(seg[1:])   # skip first (duplicate)

    for q in trajectory:
        for j in range(7):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=float(q[j]), force=control_force)
        p.stepSimulation()
        
        collided_this_step = False
        for obstacle_id, obstacle_cost in obstacles: 
            contact_points = p.getContactPoints(bodyA=robot, bodyB=obstacle_id)
            if contact_points:
                collided_this_step = True
                total_collided_obstacles.add((obstacle_id, obstacle_cost))

        for obj in scene.scene_objects: 
            closest_points = p.getClosestPoints(bodyA=robot, bodyB=obj.id, distance=10)
            for closest_point in closest_points:
                distance = closest_point[8]
                if distance < min_distance[obj.name]:
                    min_distance[obj.name] = distance

        if collided_this_step: 
            total_collision_time += step_time
        reach_target_time += step_time

        contact_points = p.getContactPoints(bodyA=robot, bodyB=scene.target_id)
        ee_contacts = [contact for contact in contact_points if contact[3] in contact_index]
        if ee_contacts:
            target_reached = True
            break
        if time.time() - start_time > MAX_DURATION:
            print("Simulation exceeded maximum duration.")
            break

    # reach_target
    target_reached = int(target_reached)

    # path_cost
    path_cost = sum(obstacle_cost for _, obstacle_cost in total_collided_obstacles)

    # displacement
    for obj in scene.scene_objects:
        init_pos = obj.position
        pos, ori = p.getBasePositionAndOrientation(obj.id)
        final_pos = np.array(pos)
        displacement = np.linalg.norm(final_pos - init_pos)
        displacements[obj.name] = displacement

    metric_data['reach_target'] = target_reached
    metric_data['path_cost'] = path_cost
    metric_data['collision_time'] = total_collision_time
    metric_data['reach_target_time'] = reach_target_time
    for obj_name, dis in min_distance.items():
        metric_data["minimum_distance:" + obj_name] = dis
    for obj_name, dis in displacements.items():
        metric_data["displacement:" + obj_name] = dis

    reset_scene(scene, robot, initial_state)

    return metric_data


def reset_scene(scene, robot, initial_state):
    reset_robot_to_initial_state(robot, initial_state)
    scene.reset_objects()

    
def reset_robot_to_initial_state(robot, initial_state):
    p.resetBasePositionAndOrientation(
        robot,
        initial_state['base_position'],
        initial_state['base_orientation']
    )

    p.resetBaseVelocity(
        robot,
        linearVelocity=initial_state['base_velocity'],
        angularVelocity=initial_state['base_angular_velocity']
    )

    num_joints = p.getNumJoints(robot)
    for i in range(num_joints):
        joint_position = initial_state['joint_states'][i][0]
        joint_velocity = initial_state['joint_states'][i][1]
        p.resetJointState(robot, i, joint_position, joint_velocity)


def compute_stats(data):
    return {
        "mean": np.mean(data).item(),   
        "variance": np.var(data).item(),
        "min": np.min(data).item(),    
        "max": np.max(data).item(),  
        "min_index": int(np.argmin(data)) + 1,
        "max_index": int(np.argmax(data)) + 1,
    }


