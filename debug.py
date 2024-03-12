import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from data_utils import get_nth_episode_info
from IPython.display import clear_output
from sapien.core import Pose

from manipulation.habitat_classes import *

from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv
from mani_skill2.utils.registration import register_env
import gymnasium as gym
import sapien.core as sapien
from mani_skill2 import ASSET_DIR
from pathlib import Path
from typing import Dict, List
from mani_skill2 import format_path
from mani_skill2.utils.io_utils import load_json


from transforms3d.euler import euler2quat, quat2euler


    
@register_env("CustomEnv-v0", max_episode_steps=200, override=True)
class CustomEnv(PickCubeEnv):
    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_cube(self.cube_half_size*4, static=True)
        self.obj2 = self._build_cube(self.cube_half_size*4, color=(0, 0, 1), static=True)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _register_cameras(self):
        pose = look_at([-1, 0, 1], [0, 0, 0])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 1024, 1024, np.pi/3, 0.01, 10
        )
    

env = gym.make("CustomEnv-v0", obs_mode="rgbd")
_ = env.reset()