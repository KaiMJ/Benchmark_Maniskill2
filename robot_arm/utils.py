import numpy as np
from mani_skill2.utils.sapien_utils import look_at
import matplotlib.pyplot as plt
import h5py

def plot_img(img, title=None):
    plt.figure(figsize=(10,6))
    if title is not None:
        plt.title(title)
    plt.imshow(img)

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out



import random


def generate_instructions(n, max_distance=4):
    # Define the directions for angle and distance
    direction_for_angle = [
        "left", "right", "up", "down", # "backward", "forward"
    ]

    direction_for_distance = [
        "left", "right", "up", "down", "backward", "forward"
    ]

    instructions = []
    min_angle = 30
    max_angle = 90
    min_distance = 0.5
    max_distance = 2

    
    for _ in range(n):
        movement_type = random.choice(['angle', 'distance'])
        
        if movement_type == 'angle':
            direction = random.choice(direction_for_angle)
            if direction != "backward" or direction != "forward":
                value = random.choice([30, 45, 60, 90])
            else:
                value = 180
        else:
            direction = random.choice(direction_for_distance)
            value = random.choice([0.25, 0.5, 0.75, 1])
        instructions.append((direction, movement_type, value))
    
    return instructions

    # # Example usage
    # n = 3  # Number of instructions to generate
    # instructions = generate_instructions(n)
    # print(instructions)

def rotate_around_axis(position, target, axis, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Direction vector from position to target
    direction = np.array(target) - np.array(position)
    
    # Rodrigues' rotation formula
    rotated_direction = (direction * np.cos(angle_radians) +
                         np.cross(axis, direction) * np.sin(angle_radians) +
                         axis * np.dot(axis, direction) * (1 - np.cos(angle_radians)))
    
    # New target position after rotation
    new_target = np.array(position) + rotated_direction
    
    return new_target

def new_target_with(position, target, direction, angle=None, distance=None):
    """
    Adjusts the position and target based on the specified direction and type (rotation or distance adjustment).

    Parameters:
    - position: The current position.
    - target: The current target.
    - direction: Direction for adjustment ("left", "right", "up", "down", "tilt_left", "tilt_right", "backward", "forward").
    - angle: Angle for rotation adjustments. Required if type is "rotation".
    - distance: Distance for movement adjustments. Required if type is "distance".

    Returns:
    - A tuple containing the new position and target after adjustment.
    """

    if angle is not None:
        # Mapping directions to their respective rotation axes and angle adjustments
        axis_map = {
            "left": ([0, 0, 1], angle),
            "right": ([0, 0, 1], -angle),
            "up": ([0, 1, 0], angle),
            "down": ([0, 1, 0], -angle),
            # "backward": ([0, 1, 0], 180),
            # "forward": ([0, 1, 0], 180)  # Assuming forward rotation is same as backward
        }

        if direction in axis_map:
            axis, angle = axis_map[direction]
            return position, rotate_around_axis(position, target, np.array(axis), angle)
        else:
            raise ValueError("Invalid direction for rotation")

    elif distance is not None:
        # Vector adjustments for each direction
        direction_map = {
            "left": ([0, -1, 0], position[0] > target[0]),
            "right": ([0, 1, 0], position[0] < target[0]),
            "up": ([0, 0, 1], True),
            "down": ([0, 0, -1], True),
            "backward": ([1, 0, 0], True),
            "forward": ([-1, 0, 0], True)
        }

        if direction in direction_map:
            vec, condition = direction_map[direction]
            vector = np.array(vec)
            # Flip the vector if condition is False
            if not condition:
                vector = -vector
            new_target = target + distance * vector
            new_position = position + distance * vector
            return new_position, new_target
        else:
            raise ValueError("Invalid direction for distance")

    else:
        raise ValueError("Invalid type")


def get_poses_from_instructions(initial_position, initial_target, instructions):
    """
        initial_position: [x, y, z]
        intial_target: [x, y, z]
        instructions: [(string direction, movement_type, value)]
    """
    poses = []
    position = initial_position
    target = initial_target
    pose = look_at(position, target)
    poses.append(pose)

    for instruction in instructions:
        direction, movement_type, value = instruction
        if movement_type == "angle":
            position, target = new_target_with(position, target, direction, angle=value)
        elif movement_type == "distance":
            position, target = new_target_with(position, target, direction, distance=value)
        else:
            raise ValueError(f"Invalid movement type: {movement_type}")

        pose = look_at(position, target)
        poses.append(pose)

    return poses

# def get_poses_from_multiple_views():




import matplotlib.pyplot as plt
import numpy as np

# Register ManiSkill2 environments in gym
import mani_skill2.envs

from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at

from mani_skill2 import ASSET_DIR
from collections import OrderedDict
import gymnasium as gym

import numpy as np
# import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv

import warnings




def load_pickcube_v1(camera_poses, offset_pose: Pose, rotation_pose: Pose, env_info, IMG_WIDTH=512, IMG_HEIGHT=512):
    """
    Get pick cube environment with the following modifications:
        - habitat scene
        - rotations
        - camera configs

    parameters:
        - camera_poses for multiple cameras
        - offset of the scene
        - rotation of the scene    
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        @register_env("PickCube-v1", max_episode_steps=200, override=True)
        class PickCubeEnv2(PickCubeEnv):
            def _load_actors(self):
                # -------------------------------------------------------------------------- #
                # Load static scene
                # -------------------------------------------------------------------------- #

                builder = self._scene.create_actor_builder()
                path = f"{ASSET_DIR}/hab2_bench_assets/stages/Baked_sc1_staging_00.glb"
                pose = Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
                # rotation_pose = sapien.Pose(q=euler2quat(0, 0, -np.pi/2)) # 90
                # pose = pose * rotation_pose
                
                # NOTE: use nonconvex collision for static scene
                builder.add_nonconvex_collision_from_file(path, pose)
                builder.add_visual_from_file(path, pose)
                self.arena = builder.build_static()
                # Add offset so that the workspace is on the table
                # offset = sapien.Pose(-np.array([-2.0616, -3.1837, 0.66467 + 0.095]))

                offset = rotation_pose * offset_pose
                self.arena.set_pose(offset)

                # self._add_ground(render=self.bg_name is None)
                self.obj = self._build_cube(self.cube_half_size)
                self.goal_site = self._build_sphere_site(self.goal_thresh)

            def _register_cameras(self):
                cameras = []
                pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
                cameras.append(CameraConfig(
                    "base_camera", pose.p, pose.q, IMG_WIDTH, IMG_HEIGHT, np.pi / 2, 0.01, 10
                ))

                for i in range(len(camera_poses)):
                    scene_camera = CameraConfig(
                        f"camera_{i}", camera_poses[i].p, camera_poses[i].q, IMG_WIDTH, IMG_HEIGHT, np.pi / 2, 0.01, 10
                    )
                    cameras.append(scene_camera)
                return cameras

        env = gym.make("PickCube-v1", camera_cfgs={"add_segmentation": True}, **env_info["env_kwargs"])
    return env
