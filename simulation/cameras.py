import numpy as np
import pybullet as p


class Camera:
    image_size = (800, 800)
    height, width = image_size
    intrinsics = np.array([[500, 0, 400], [0, 500, 400], [0, 0, 1]])

    z_range = (0.01, 10.0)
    target_position = [0, 0, 0]

    def __init__(self, name, camera_position, camera_rotation):
        self.name = name
        self.position = camera_position
        
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rot_matrix = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(camera_rotation))
        self.rotation = np.float32(rot_matrix).reshape(3, 3)

        lookdir = (self.rotation @ lookdir).reshape(-1)
        updir = (self.rotation @ updir).reshape(-1)
        lookat = self.target_position + lookdir
        self.view_matrix = p.computeViewMatrix(self.position, lookat, updir)

        fx = self.intrinsics[0][0]
        znear, zfar = self.z_range
        fovh = (self.height / 2) / fx
        fovh = 180 * np.arctan(fovh) * 2 / np.pi
        aspect_ratio = self.height / self.width
        self.projection_matrix = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)
