import numpy as np
from sapien.core import Pose
import matplotlib.pyplot as plt

# Register ManiSkill2 environments in gym
import mani_skill2.envs
from mani_skill2 import ASSET_DIR
from mani_skill2.utils.sapien_utils import look_at


from mani_skill2.utils.registration import register_env

from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.envs.pick_and_place.stack_cube import StackCubeEnv
from mani_skill2.envs.pick_and_place.pick_single import PickSingleYCBEnv
from mani_skill2.envs.pick_and_place.pick_clutter import PickClutterYCBEnv
from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv
from mani_skill2.envs.assembly.plug_charger import PlugChargerEnv
from mani_skill2.envs.assembly.assembling_kits import AssemblingKitsEnv
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.envs.ms1.push_chair import PushChairEnv
from mani_skill2.envs.ms1.open_cabinet_door_drawer import OpenCabinetDoorEnv, OpenCabinetDrawerEnv
from mani_skill2.envs.ms1.move_bucket import MoveBucketEnv

from mani_skill2.utils.sapien_utils import hex2rgba





# from tqdm import tqdm
# import numpy as np
# from utils import load_pickcube_v1, get_poses_from_instructions
# import h5py, json
# from mani_skill2.utils.io_utils import load_json
# from skimage import io, color, filters
# from transforms3d.euler import euler2quat

# from sapien.core import Pose


# import matplotlib.pyplot as plt
# import numpy as np

# # Register ManiSkill2 environments in gym
# import mani_skill2.envs

# # from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.sensors.camera import CameraConfig
# from mani_skill2.utils.registration import register_env
# from mani_skill2.utils.sapien_utils import look_at

# from mani_skill2 import ASSET_DIR
# from collections import OrderedDict
# import gymnasium as gym

# import numpy as np
# # import sapien.core as sapien
# from sapien.core import Pose
# from transforms3d.euler import euler2quat, quat2euler

# from mani_skill2.utils.registration import register_env
# from mani_skill2.utils.sapien_utils import vectorize_pose
# import warnings


def add_scene(self, env_id=None):
    path = f"{ASSET_DIR}/hab2_bench_assets/stages/Baked_sc1_staging_00.glb"
    pose = Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes

    builder = self._scene.create_actor_builder()
    # builder.add_nonconvex_collision_from_file(path, pose)
    builder.add_visual_from_file(path, pose)
    self.arena = builder.build_static(name="visual_ground")

    # offset_lim = 1.5
    # offset_pose = sapien.Pose(-np.array([np.random.uniform(-offset_lim, offset_lim), 0, 0]))
    # rotation_pose = sapien.Pose(q=euler2quat(0, 0, np.random.uniform(0, 2*np.pi)))
    # offset = offset_pose * rotation_pose
    # self.arena.set_pose(offset)

def get_poses(N, env_id=None):
    # camera_poses = [look_at((0.4, 0.4, 0.8), (0, 0, 0.4))]
    camera_poses = []

    # pick_cube base camera
    initial_point = np.array([0.3, 0, 0.6])
    target_point = np.array([-0.1, 0, 0.1])
    fov = np.pi / 2
    # fov = 1

    # # pick cube panda render
    # initial_point = [0.4, 0.4, 0.8]
    # target_point = np.array([0, 0, 0.4])
    # fov = 1

    radius = np.sqrt(initial_point[0]**2 + initial_point[1]**2) # Radius of the circle at Z = 0.8
    z_height = initial_point[2] / 3

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Calculate the X, Y coordinates of the points on the circle at Z = 0.8
    x_points = radius * np.cos(angles)
    y_points = radius * np.sin(angles)
    z_points = np.full(N, z_height)

    for i in range(N):
        camera_poses.append(look_at((x_points[i], y_points[i], z_points[i]), target_point))
    return camera_poses, fov

IMG_WIDTH = 512
IMG_HEIGHT = 512

def register_cameras(self, env_id):
    cameras = []
    camera_poses, fov = get_poses(9, env_id)

    for i in range(len(camera_poses)):
        scene_camera = CameraConfig(
            f"camera_{i}", camera_poses[i].p, camera_poses[i].q, IMG_WIDTH, IMG_HEIGHT, fov, 0.01, 10
        )
        cameras.append(scene_camera)
    return cameras


"""
Pick and Place
"""

apply_scene = np.random.choice([True, False])
apply_scene = False


@register_env("PickCube-v0", max_episode_steps=200, override=True)
class PickCubeEnv2(PickCubeEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "PickCube-v0")

        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _register_cameras(self):
        return register_cameras(self, env_id="PickCube-v0")

@register_env("StackCube-v0", max_episode_steps=200, override=True)
class StackCubeEnv2(StackCubeEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "StackCube-v0")

        self.box_half_size = np.float32([0.02] * 3)
        self.cubeA = self._build_cube(self.box_half_size, color=(1, 0, 0), name="cubeA")
        self.cubeB = self._build_cube(
            self.box_half_size, color=(0, 1, 0), name="cubeB", static=False
        )

    def _register_cameras(self):
        return register_cameras(self, env_id="StackCube-v0")

@register_env("PickSingleYCB-v0", max_episode_steps=200, override=True)
class PickSingleYCBEnv2(PickSingleYCBEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "PickSingleYCB-v0")

        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _register_cameras(self):
        return register_cameras(self, env_id="PickSingle-v0")

@register_env("PickClutterYCB-v0", max_episode_steps=200, override=True)
class PickClutterYCBEnv2(PickClutterYCBEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "PickSingleEGAD-v0")

        self.objs = []
        self.bbox_sizes = []
        for actor_cfg in self.episode["actors"]:
            model_id = actor_cfg["model_id"]
            model_scale = actor_cfg["scale"]
            obj = self._load_model(model_id, model_scale=model_scale)
            self.objs.append(obj)

            bbox = self.model_db[model_id]["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.bbox_sizes.append(bbox_size * model_scale)

        self.target_site = self._build_sphere_site(
            0.01, color=(1, 1, 0), name="_target_site"
        )
        self.goal_site = self._build_sphere_site(
            0.01, color=(0, 1, 0), name="_goal_site"
        )

    def _register_cameras(self):
        return register_cameras(self, env_id="PickClutter-v0")

"""
Assembly
"""

@register_env("PegInsertionSide-v0", max_episode_steps=200, override=True)
class PegInsertionSideEnv2(PegInsertionSideEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "PegInsertionSide-v0")

        # peg
        # length, radius = 0.1, 0.02
        length = self._episode_rng.uniform(0.075, 0.125)
        radius = self._episode_rng.uniform(0.015, 0.025)
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = 0.5 * (length - radius) * self._episode_rng.uniform(-1, 1, size=2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius


@register_env("PlugCharger-v0", max_episode_steps=200, override=True)
class PlugChargerEnv2(PlugChargerEnv):    
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "PegInsertionSide-v0")

        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

@register_env("AssemblingKits-v0", max_episode_steps=200, override=True)
class AssemblingKitsEnv2(AssemblingKitsEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "AssemblingKits-v0")

        self.kit = self._load_kit()
        self.obj = self._load_object(self.object_id)
        self._other_objs = [self._load_object(i) for i in self._other_objects_id]


"""
Miscellaneous
"""

@register_env("TurnFaucet-v0", max_episode_steps=200, override=True)
class TurnFaucetEnv2(TurnFaucetEnv):
    def _load_actors(self):
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, "TurnFaucet-v0")

"""
Mobile Manipulation
"""

def load_ms1_actors(self, env_id=None):
    # Create a collision ground plane
        ground = self._add_ground(render=self.bg_name is None)
        if apply_scene:
            ground.set_name("ignore_ground")
            add_scene(self, env_id)

        # Specify a collision (ignore) group to avoid collision with robot torso
        cs = ground.get_collision_shapes()[0]
        cg = cs.get_collision_groups()
        cg[2] = cg[2] | 1 << 30
        cs.set_collision_groups(*cg)

        if self.bg_name is None:
            # Create a visual ground box
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
            builder = self._scene.create_actor_builder()
            builder.add_box_visual(
                pose=Pose([0, 0, -1]), half_size=[50, 50, 1], material=rend_mtl
            )
            visual_ground = builder.build_static(name="visual_ground")

@register_env("OpenCabinetDoor-v1", max_episode_steps=200, override=True)
class OpenCabinetDoorEnv2(OpenCabinetDoorEnv):
    def _load_actors(self):
        load_ms1_actors(self, "OpenCabinetDoor-v1")

@register_env("OpenCabinetDrawer-v1", max_episode_steps=200, override=True)
class OpenCabinetDrawerEnv2(OpenCabinetDrawerEnv):
    def _load_actors(self):
        load_ms1_actors(self, "OpenCabinetDrawer-v1")

@register_env("PushChair-v1", max_episode_steps=200, override=True)
class PushChairEnv2(PushChairEnv):
    def _load_actors(self):
        load_ms1_actors(self, "PushChair-v1")

        # A red sphere to indicate the target to push the chair.
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=0.15, color=(1, 0, 0))
        self.target_indicator = builder.build_static(name="target_indicator")


    # def _register_cameras(self):
    #     return register_cameras(self, env_id="PushChair-v1")


@register_env("MoveBucket-v1", max_episode_steps=200, override=True)
class MoveBucketEnv2(MoveBucketEnv):
    def _load_actors(self):
        load_ms1_actors(self, "MoveBucket-v1")

        # place a target platform on ground
        box_half_size = [0.3, 0.3, 0.1]
        builder = self._scene.create_actor_builder()

        white_diffuse = self._renderer.create_material()
        white_diffuse.base_color = [0.8, 0.8, 0.8, 1]
        white_diffuse.metallic = 0.0
        white_diffuse.roughness = 0.9
        white_diffuse.specular = 0
        builder.add_box_visual(half_size=box_half_size, material=white_diffuse)

        obj_material = self._scene.create_physical_material(0.5, 0.5, 0)
        builder.add_box_collision(
            half_size=box_half_size, material=obj_material, density=1000
        )
        self.target_platform = builder.build_static(name="target_platform")

        # balls
        R = 0.05
        self.balls_radius = R
        builder = self._scene.create_actor_builder()
        builder.add_sphere_collision(radius=R, density=1000)
        builder.add_sphere_visual(radius=R, color=[0, 1, 1])
        self.balls = []
        self.GX = self.GY = self.GZ = 1
        for i in range(self.GX * self.GY * self.GZ):
            actor = builder.build(name="ball_{:d}".format(i + 1))
            self.balls.append(actor)
