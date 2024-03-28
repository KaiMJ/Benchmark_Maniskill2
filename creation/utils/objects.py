from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from pathlib import Path
import gymnasium as gym
from sapien.core import Pose
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from utils.config import ycb_heights, color_maps
from glob import glob
from scipy import sparse
from PIL import Image
import json
import os

class GenerationClass:
    def __init__(self, model_paths=None):
        self.env = None
        self.configs = None
        self.camera_cfgs = {"p": [-1, 0, 1], "fov": 1.4}
        self.low = -0.5
        self.high = 0.5
        self.hoirzontally = False
        self.init = False

        if model_paths is None:
            model_paths = sorted(glob(
                "../data/mani_skill2_ycb/models/*"), key=lambda x: int(x.split('/')[-1][:3]))
        self.model_paths = model_paths

    # ================== Initializing environment ==================

    def get_env(self, configs, camera_cfgs=None):
        self.configs = configs
        if camera_cfgs is not None:
            self.camera_cfgs = camera_cfgs

        self.register_objects(self.camera_cfgs)
        self.env = gym.make("CustomEnv-v0", obs_mode="rgbd",
                            camera_cfgs={"add_segmentation": True})
        self.env.reset()
        self.env.unwrapped.get_articulations()[0].set_pose(Pose([0, 0, -2], [1, 0, 0, 0]))
        return self.env

    def register_objects(self, camera_cfgs):
        """
            Configs: dict[
                "name": "cube" / "sphere" / "002_master_chef_can"
                "scale": float or none
                "color": tuple or none
            ]
        """

        def build_asset(self, config):
            scale = config["scale"]
            if config["name"].split('_')[-1] == "cube":
                obj = self._build_cube(
                    self.cube_half_size*scale, color=config["color"], static=config["static"])
                obj.name = config["name"]
            else:
                scale *= self.cube_half_size / 0.01887479572529618
                filepath = config["name"]
                model_dir = Path(filepath).resolve()
                # model_dir = ".." / ASSET_DIR / "mani_skill2_ycb/models/011_banana"
                builder = self._scene.create_actor_builder()

                collision_file = str(model_dir / "collision.obj")
                builder.add_multiple_collisions_from_file(
                    filename=collision_file, scale=scale, density=1000
                )
                visual_file = str(model_dir / "textured.obj")
                builder.add_visual_from_file(filename=visual_file, scale=scale)
                # builder.add_box_visual()
                if config["static"]:
                    obj = builder.build_static(name=filepath.split("/")[-1])
                else:
                    obj = builder.build(name=filepath.split("/")[-1])
                    obj.lock_motion(False, False, False, True, True, True)

            return obj

        configs = self.configs
        @register_env("CustomEnv-v0", max_episode_steps=200, override=True)
        class CustomEnv(PickCubeEnv):
            def _load_actors(self):
                self._add_ground(render=self.bg_name is None)
                objs = []
                for i, config in enumerate(configs):
                    obj = build_asset(self, config)
                    if i == 0:
                        self.obj = obj
                    objs.append(obj)

                self.objs = objs
                self.goal_site = self._build_sphere_site(self.goal_thresh)

            def _register_cameras(self):
                pose = look_at(camera_cfgs["p"], [0, 0, 0])
                return CameraConfig(
                    "base_camera", pose.p, pose.q, 512, 512, camera_cfgs["fov"], 0.01, 10
                )

    # ================== Initializing object positions ==================

    def is_overlapping(self, pos1, pos2, min_dist):
        # Calculate the Euclidean distance between two positions
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return dist < min_dist

    def generate_non_overlapping_position(self, existing_positions, min_dist, low=-0.5, high=0.5):
        while True:
            # Generate a random position
            new_position = np.random.uniform(low=low, high=high, size=(2))
            # Check for overlap with existing positions
            overlap = any(self.is_overlapping(new_position, pos, min_dist)
                          for pos in existing_positions)
            if not overlap:
                return new_position

    def initialize_random_positions(self, low=-0.5, high=0.5, min_dist=0.5, hoirzontally=False):
        existing_positions = []
        if hoirzontally:
            positions = np.linspace(low, high, len(self.configs))

        for move_idx in range(len(self.configs)):
            scale = self.configs[move_idx]["scale"]
            if hoirzontally:
                position = positions[move_idx]
                position = np.array([position, 0])
            else:
                position = self.generate_non_overlapping_position(
                    existing_positions, min_dist, low, high)
                existing_positions.append(position)

            quat = euler2quat(0, 0, np.random.uniform(-np.pi*2, np.pi*2))
            name = self.configs[move_idx]["name"]
            if name.split('_')[-1] == "cube":
                new_pose = Pose(
                    [position[0], position[1], self.env.unwrapped.cube_half_size[-1]*scale], quat)
            else:
                model_name = name.split('/')[-1]
                scale = self.configs[move_idx]["scale"]
                new_pose = Pose([position[0], position[1],
                                ycb_heights[model_name]*scale*1.02], quat)

            self.env.unwrapped.objs[move_idx].set_pose(new_pose)

    # ================== Placing objects in position ==================

    def place_object_in_direction(self, obj, direction, distance=0.4):
        pose_obj = obj.get_pose()
        if direction == "forward":
            new_position_obj = [pose_obj.p[0] -
                                distance, pose_obj.p[1], pose_obj.p[2]]
        elif direction == "backward":
            new_position_obj = [pose_obj.p[0] +
                                distance, pose_obj.p[1], pose_obj.p[2]]
        elif direction == "right":
            new_position_obj = [pose_obj.p[0],
                                pose_obj.p[1] - distance, pose_obj.p[2]]
        elif direction == "left":
            new_position_obj = [pose_obj.p[0],
                                pose_obj.p[1] + distance, pose_obj.p[2]]

        obj.set_pose(Pose(new_position_obj, obj.get_pose().q))

    def place_object_on_another(self, obj1, obj2, direction, distance=0.4):
        pose_obj2 = obj2.get_pose()
        pose_obj1 = obj1.get_pose()

        if direction == "front":
            new_position_obj1 = [pose_obj2.p[0] -
                                 distance, pose_obj2.p[1], pose_obj2.p[2]]
        elif direction == "behind":
            new_position_obj1 = [pose_obj2.p[0] +
                                 distance, pose_obj2.p[1], pose_obj2.p[2]]
        elif direction == "right":
            new_position_obj1 = [pose_obj2.p[0],
                                 pose_obj2.p[1] - distance, pose_obj2.p[2]]
        elif direction == "left":
            new_position_obj1 = [pose_obj2.p[0],
                                 pose_obj2.p[1] + distance, pose_obj2.p[2]]
        elif direction == "top":
            new_position_obj1 = [
                pose_obj2.p[0], pose_obj2.p[1], pose_obj2.p[2]*2 + pose_obj1.p[2]]

        obj1.set_pose(Pose(new_position_obj1, obj1.get_pose().q))

    def place_object_in_between(self, obj_to_move, obj1, obj2):
        pose_obj1 = obj1.get_pose().p
        pose_obj2 = obj2.get_pose().p

        new_position = Pose((pose_obj1 + pose_obj2) /
                            2, obj_to_move.get_pose().q)

        obj_to_move.set_pose(new_position)

    def remove_object(self, obj_to_move):
        obj_to_move.set_pose(Pose([0, 0, 10], [1, 0, 0, 0]))


    # ================== Order by color / size / name ==================

    def order_by_color(self, order_list, colors):
        objs = self.env.unwrapped.objs
        y_position = np.linspace(-0.5, 0.5, len(order_list))
        for i in range(len(order_list)):
            idx = order_list.index(colors[i])
            obj_pose = objs[idx].get_pose()
            objs[idx].set_pose(
                Pose([0, y_position[idx], obj_pose.p[2]], obj_pose.q))

    def order_by_size(self, ascending=False):
        objs = self.env.unwrapped.objs
        y_position = np.linspace(-0.5, 0.5, len(self.configs))

        sorted_size = list(np.argsort(
            [self.configs[i]["scale"] for i in range(len(objs))]))
        if ascending:
            sorted_size = sorted_size[::-1]
        for i in range(len(self.configs)):
            idx = sorted_size.index(i)
            obj_pose = objs[i].get_pose()
            objs[i].set_pose(
                Pose([0, y_position[idx], obj_pose.p[2]], obj_pose.q))

    def order_by_name(self, name_list, configs):
        objs = self.env.unwrapped.objs
        y_position = np.linspace(-0.5, 0.5, len(name_list))
        for i in range(len(name_list)):
            idx = name_list.index(configs[i]["name"])
            obj_pose = objs[idx].get_pose()
            objs[idx].set_pose(
                Pose([0, y_position[idx], obj_pose.p[2]], obj_pose.q))

    # ================== Motions ==================

    def get_random_force(self, random_vector=None):
        if random_vector is None:
            random_vector = np.random.randn(2)
        normalized_vector = random_vector / np.linalg.norm(random_vector)
        magnitude = np.sqrt(250**2 + 250**2)
        final_vector = normalized_vector * magnitude
        result = np.concatenate([final_vector, [0]])
        return result

    def move_object(self, obj, direction, motion_map, force_scale=1):
        force_scale *= obj.mass * 4
        obj.add_force_at_point(
            motion_map[direction]*force_scale, np.array([0, 0, 0]))

    def get_object_to_object_direction(self, obj_to_move, obj_target):
        pose_obj = obj_to_move.get_pose()
        pose_target = obj_target.get_pose()
        diff = pose_obj.p[:2] - pose_target.p[:2]
        magnitude = np.linalg.norm(diff)
        direction = diff / magnitude
        direction = np.concatenate([direction, [0]])
        return -direction

    def move_object_toward_another(self, obj_to_move, direction, force_scale=1):
        force_scale *= 250 * obj_to_move.mass * 4
        obj_to_move.add_force_at_point(
            direction * force_scale, np.array([0, 0, 0]))

    # ================== Configurations ==================

    def cube_configs(self, n, colors=None, scales=None, static=True):
        colors = np.random.choice(list(color_maps.keys()), 4, replace=False)

        if scales is not None:
            random_scales = scales
        else:
            random_scales = np.random.permutation(np.arange(3, 7))

        obj_configs = []
        for i in range(n):
            color = colors[i]
            color_value = color_maps[color]
            obj_configs.append({
                "name": f"{color}_cube",
                "scale": random_scales[i],
                "color": color_value,
                "color_name": color,
                "static": static,
            })

        camera_cfgs = {"p": [-1, 0, 1], "fov": 1.3}

        return obj_configs, camera_cfgs

    def ycb_configs(self, n):
        configs = [
            {
                "name": self.model_paths[i],
                "scale": 2,
                "static": True,
            }
            for i in np.random.choice(np.arange(len(self.model_paths)), n, replace=False).astype(int)
        ]
        # if n == 1:
        camera_cfgs = {"p": [-1, 0, 1], "fov": 1.2}

        return configs, camera_cfgs  


    # ================== Save JSON ==================
    def init_dir(self, data_dir):
        if not self.init:
            for name in ["initial", "result", "initial_seg", "result_seg"]:
                os.makedirs(os.path.join(data_dir, name), exist_ok=True)
                self.init = True
                self.data_dir = data_dir

    def save_json(self, json_data, initial_img, result_img, seg_before, seg_after):
        data_dir = self.data_dir

        Image.fromarray(initial_img).save(os.path.join(data_dir, f"{json_data['initial_img']}"))
        Image.fromarray(result_img).save(os.path.join(data_dir, f"{json_data['result_img']}"))
        sparse.save_npz(os.path.join(data_dir, f"{json_data['initial_seg']}"), sparse.csr_matrix(seg_before))
        sparse.save_npz(os.path.join(data_dir, f"{json_data['result_seg']}"), sparse.csr_matrix(seg_after))

        # save json
        json_path = os.path.join(data_dir, f"data.json")
        try:
            current_data = json.load(open(json_path, "r"))
        except:
            current_data = []
        with open(json_path, "w") as f:
            json.dump(current_data + [json_data], f)

    # ================== Get Segmentation Masks ==================

    def get_seg_masks(self, obs):
        seg = obs['image']['base_camera']['Segmentation'][..., :3]
        object_ids = np.array([obj.id - 2 for obj in self.env.unwrapped.objs])
        object_names = np.array([obj.name for obj in self.env.unwrapped.objs])

        mask = np.isin(seg, object_ids)

        seg_masked = np.where(mask, seg, 0)

        starting_id = min(object_ids) - 1
        object_ids_adjusted = (object_ids - starting_id).astype(str)

        seg_masked = np.where(seg_masked > 0, seg_masked - starting_id, 0)
        # seg_masked = np.amax(seg_masked, axis=2)
        return seg_masked[:, :, 0], list(zip(object_names, object_ids_adjusted))
