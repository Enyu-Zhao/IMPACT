import pybullet as p
import pybullet_data
import json
import numpy as np

from simulation.cameras import Camera
from simulation.image_set import ImageSet
from simulation.cost_map import CostMap, fetch_cost_dict_bypass


class SceneObject:
    def __init__(self, object_name, position, orientation=[0, 0, 0], scale=1, cost=0, scene=None):
        self.object_name = object_name
        self.name = object_name
        self.asset_name = object_name + "_textured"
        self.asset_path = f"assets/objects/{self.object_name}"
        self.file_path = f"{self.asset_path}/{self.object_name}.urdf"
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation) 
        self.scale = scale
        self.cost = cost
        self.id = None

        with open(self.file_path, "w") as file:
            file.write(
                f"""<?xml version="1.0" ?>
<robot name="{self.object_name}">
    <link name="baseLink">
        <contact>
        <friction_anchor/>
        <lateral_friction value="1.5"/>
        <rolling_friction value="0.1"/>
        <spinning_friction value="0.1"/>
        <contact_cfm value="1e-5"/>
        <contact_erp value="0.8"/>
        </contact>
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{self.asset_path}/meshes/{self.asset_name}.obj" scale="1 1 1"/>
            </geometry>
            <material name="white">
                <color rgba="1 1 1 1"/>
            </material>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{self.asset_path}/meshes/{self.asset_name}.obj" scale="1 1 1"/>
            </geometry>
        </collision>
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
    </link>
</robot>"""
            )


class Scene:
    def __init__(self, name="scene", scene_data=None):
        self.name = name
        self.scene_data = scene_data

        object_models = scene_data["object_models"]
        self.scene_objects = [
            SceneObject(
                object_name,
                position=obj_data["position"],
                orientation=obj_data["orientation"] if "orientation" in obj_data else [0, 0, 0],
                scale=obj_data["scale"],   
                cost=obj_data["cost"]       
            ) 
            for object_name, obj_data in object_models.items()]

        self.set_cameras()


    def set_cameras(self):
        front_camera = Camera("front_cam", [1, 0, 1], [np.pi / 4, np.pi, -np.pi / 2])
        left_camera = Camera("left_cam", [0, -1, 1], [np.pi / 4, np.pi, np.pi])
        right_camera = Camera("right_cam", [0, 1, 1], [np.pi / 4, np.pi, 0])
        self.cameras = [front_camera, left_camera, right_camera]


    def load_robot(self):
        robot_base_location = [0.6, 0, 0]
        robot = p.loadURDF("franka_panda/panda.urdf", robot_base_location, p.getQuaternionFromEuler([0, 0, np.pi]), useFixedBase=True)
        ee_link_index = 11 
        self.initial_joint_positions = [0.05, -1.15, -0.13, -3.06, -0.05, 2.05, -2.35]
        for i in range(len(self.initial_joint_positions)):
            p.resetJointState(robot, i, self.initial_joint_positions[i])
            
        p.resetBasePositionAndOrientation(robot, robot_base_location, p.getQuaternionFromEuler([0, 0, np.pi]))
        p.changeDynamics(robot, -1, contactProcessingThreshold=0.0, linearDamping=0.1, angularDamping=0.1)

        return robot, ee_link_index
    
    
    def get_target_object_info(self, target_object_name):
        target_id = None
        target_position = None
        for obj in self.scene_objects:
            if obj.name == target_object_name:
                target_id = obj.id
                base_position = obj.position  
                aabb_min, aabb_max = p.getAABB(target_id)
                object_top_z = aabb_max[2]  
                target_position = [base_position[0], base_position[1], object_top_z]
                break
        if target_position is None:
            raise ValueError(f"Target object '{target_object_name}' not found in the scene.")
        
        self.target_id = target_id
        self.target_position = target_position
        return target_id, target_position


    def start_physics_client(self, gui=True):
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setTimeStep(1/240)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.base_id = p.loadSDF(f"assets/objects/table_top/table_top.sdf")[0]

        num_joints = p.getNumJoints(self.base_id)
        p.changeDynamics(self.base_id, -1, mass=10000)
        for joint_index in range(num_joints):
            p.changeDynamics(self.base_id, joint_index, mass=10000)

        p.createConstraint(
            parentBodyUniqueId=self.plane_id,
            parentLinkIndex=-1, 
            childBodyUniqueId=self.base_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -1],  
            childFramePosition=[0, 0, 0],  
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.resetDebugVisualizerCamera(2, 90, -30, [0, 0, 0.5], self.physics_client)


    def load_model(self, scene_object):
        if scene_object not in self.scene_objects:
            self.scene_objects.append(scene_object)

        scene_object.id = p.loadURDF(
            scene_object.file_path,
            scene_object.position,
            scene_object.orientation,
            globalScaling=scene_object.scale,
            physicsClientId=self.physics_client,
        )
        

    def load_all_models(self):
        for i, scene_object in enumerate(self.scene_objects):
            print(f"Loading scene object {i+2}: {scene_object.name}")
            self.load_model(scene_object)

        for scene_object in self.scene_objects:
            p.changeDynamics(scene_object.id, -1, mass=100)


    def reset_objects(self):
        for scene_object in self.scene_objects:
            p.resetBasePositionAndOrientation(scene_object.id, scene_object.position, scene_object.orientation)


    def capture_image(self, show_plot=False):
        image_sets = []
        for camera in self.cameras:
            arrays = p.getCameraImage(
                width=camera.width,
                height=camera.height,
                viewMatrix=camera.view_matrix,
                projectionMatrix=camera.projection_matrix,
            )[2:5]

            image_set = ImageSet(scene=self, arrays=arrays, camera=camera)
            image_sets.append(image_set)
 
            if camera.name == "front_cam": # used for cost map generation
                self.front_camera_image_set = image_set

            if show_plot:
                image_set.show()

        return image_sets
    
    
    def get_obstacles(self):
        obstacles = [(obj.id, self.cost_dict[obj.id]) for obj in self.scene_objects if obj.id in self.cost_dict and obj.id != self.target_id]
        obstacles.append((self.base_id, self.cost_dict[self.base_id]))
        return obstacles
    
    
    def update_scene_objects(self, scene_file):
        for obj in self.scene_objects:
            obj.cost = self.cost_dict[obj.id]

        object_models = self.scene_data["object_models"]
        for obj in self.scene_objects:
            object_models[obj.name]["cost"] = obj.cost
        
        with open(scene_file, "w") as f:
            json.dump(self.scene_data, f, indent=4)


    def generate_cost_map(self, scene_file=None, target_object_name=None, show_annotated=False):
        image_sets = self.capture_image()
        for image_set in image_sets:
            if show_annotated:
                image_set.show("annotated")

        # self.cost_dict = fetch_cost_dict(image_set=self.front_camera_image_set)
        self.cost_dict = fetch_cost_dict_bypass(image_set=self.front_camera_image_set)
        print(f"[INFO]: cost_dict: {self.cost_dict}")

        self.cost_dict={2:2, 3:8, 4:6}
        self.get_target_object_info(target_object_name)
        self.update_scene_objects(scene_file)
        self.cost_dict[self.target_id] = -1
        self.cost_dict[self.base_id] = 10
        self.cost_dict[self.plane_id] = 10

        cost_map = CostMap(self.name, image_sets=image_sets, cost_dict=self.cost_dict)
        return cost_map
    

    def load_cost_map(self, target_object_name=None):
        self.cost_dict = {obj.id: obj.cost for obj in self.scene_objects}
        self.get_target_object_info(target_object_name)
        self.cost_dict[self.target_id] = 0
        self.cost_dict[self.base_id] = 10
        self.cost_dict[self.plane_id] = 10

