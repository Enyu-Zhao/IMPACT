import math
import pybullet as p
import json
import argparse

from simulation.scene import Scene


def main(args):
    scene_file = f"assets/scenes/{args.scene}/scene.json"
    with open(scene_file, 'r') as f:
        scene_data = json.load(f)

    scene = Scene(scene_data=scene_data, name=args.scene)
    scene.start_physics_client(gui=True)
    scene.load_all_models()
    scene.load_robot()

    cam_dist   = 1.2
    cam_yaw    = 45.0
    cam_pitch  = -30.0
    cam_target = [0.0, 0.0, 0.3]

    p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)
    p.setRealTimeSimulation(0)

    print("Scene loaded. Keyboard controls:")
    print("  Arrow Left/Right : orbit yaw")
    print("  Arrow Up/Down    : orbit pitch")
    print("  W/S              : pan forward/back")
    print("  A/D              : pan left/right")
    print("  +/-              : zoom in/out")
    print("  q / Ctrl+C       : exit")

    ROT_SPEED  = 0.002
    PAN_SPEED  = 0.002
    ZOOM_SPEED = 0.1

    try:
        while True:
            p.stepSimulation()
            keys = p.getKeyboardEvents()

            if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
                break

            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                cam_yaw -= ROT_SPEED
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                cam_yaw += ROT_SPEED
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                cam_pitch = min(89, cam_pitch + ROT_SPEED)
            if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                cam_pitch = max(-89, cam_pitch - ROT_SPEED)

            rad = math.radians(cam_yaw)
            if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
                cam_target[0] += math.cos(rad) * PAN_SPEED
                cam_target[1] += math.sin(rad) * PAN_SPEED
            if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
                cam_target[0] -= math.cos(rad) * PAN_SPEED
                cam_target[1] -= math.sin(rad) * PAN_SPEED
            if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
                cam_target[0] -= math.sin(rad) * PAN_SPEED
                cam_target[1] += math.cos(rad) * PAN_SPEED
            if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
                cam_target[0] += math.sin(rad) * PAN_SPEED
                cam_target[1] -= math.cos(rad) * PAN_SPEED

            if ord('=') in keys and keys[ord('=')] & p.KEY_IS_DOWN:
                cam_dist = max(0.2, cam_dist - ZOOM_SPEED)
            if ord('-') in keys and keys[ord('-')] & p.KEY_IS_DOWN:
                cam_dist = min(10.0, cam_dist + ZOOM_SPEED)

            p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)

    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="scene01")
    args = parser.parse_args()
    main(args)
