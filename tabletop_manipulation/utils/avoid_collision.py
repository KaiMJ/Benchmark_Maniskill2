import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from utils.objects import GenerationClass
from utils.config import color_maps, ycb_heights
import os
from glob import glob
from scipy import sparse
from PIL import Image
import json
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from pathlib import Path
import gymnasium as gym
from sapien.core import Pose
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
# from utils.config import ycb_heights, color_maps
from glob import glob
from scipy import sparse
from PIL import Image
import json
import os

color_maps = {
    # "red": (1, 0, 0),
    "blue": (0, 0, 1),
    "green": (0, 1, 0),
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "yellow": (1, 1, 0),
    "orange": (1, 0.5, 0),
    "purple": (0.5, 0, 0.5),
    "gray": (0.5, 0.5, 0.5),
}

class AvoidCollision():
    def __init__(self, obstacle_num=15, obj_num=4, scale=4):
        self.obstacle_num = obstacle_num
        self.obj_num = obj_num
        self.scale = scale
        
    def create_env(self, obj_num, obstacle_num=15, scale=4):
        @register_env("CustomEnv-v0", max_episode_steps=200, override=True)
        class CustomEnv(PickCubeEnv):
            def _load_actors(self):
                self._add_ground(render=self.bg_name is None)

                self.objs = []
                colors = np.random.choice(
                    list(color_maps.keys()), obj_num, replace=False)
                for i in range(obj_num):
                    color = color_maps[colors[i]]
                    if i == 0:
                        obj = self._build_cube(
                            self.cube_half_size*scale, color=color, static=True)
                        obj.name = colors[i]
                        self.obj = obj
                    else:
                        obj = self._build_cube(
                            self.cube_half_size*scale, color=color, static=True)
                        obj.name = colors[i]
                    self.objs.append(obj)

                self.obstacles = [self._build_cube(
                    self.cube_half_size*2, color=(1, 0, 0), static=True) for i in range(obstacle_num)]

                self.goal_site = self._build_sphere_site(self.goal_thresh)

            def _register_cameras(self):
                pose = look_at([-1, 0, 1], [0, 0, 0])
                return CameraConfig(
                    "base_camera", pose.p, pose.q, 512, 512, 1.7, 0.01, 10
                )
        env = gym.make("CustomEnv-v0", obs_mode="rgbd")
        env.reset()
        env.unwrapped.get_articulations()[0].set_pose(
            Pose([-1, 0, -2], [1, 0, 0, 0]))
        self.env = env
        self.unit = env.cube_half_size[0]
        self.cube_size = self.unit * scale
        return env

    def initialize_env(self):
        # Initial obstacle position and orientation
        obstacle_initial_x = np.random.randn() * 12 * self.unit
        obstacle_initial_y = np.random.randn() * 12 * self.unit
        angle = np.random.uniform(0, 2*np.pi)  # Random angle

        # Initialize the list of existing positions
        existing_positions = []

        # initialize positions of obstacles
        for i, obj in enumerate(self.env.unwrapped.obstacles):
            x = obstacle_initial_x + (i-len(self.env.unwrapped.obstacles)/2) * self.cube_size * np.cos(angle)
            y = obstacle_initial_y + (i-len(self.env.unwrapped.obstacles)/2) * self.cube_size * np.sin(angle)
            existing_positions.append([x, y, "obstacle"])
            obj.set_pose(Pose([x, y, self.unit*2], euler2quat(0, 0, angle)))

        # Function to check if a point is above the dividing line
        def is_above_line(x, y, line_angle, pivot_x, pivot_y):
            # Calculate the y-coordinate on the line for the given x
            y_on_line = np.tan(line_angle) * (x - pivot_x) + pivot_y
            return y > y_on_line

        # Function to generate a random position for a cube
        def generate_cube_position(above,obj, pivot_x, pivot_y, angle, existing_positions, cube_size):
            for _ in range(1000):  # Max attempts to find a valid position
                x = np.random.uniform(-45, 45) * self.unit
                y = np.random.uniform(-45, 45) * self.unit
                # Check if the position is on the desired side of the line and away from existing positions
                if is_above_line(x, y, angle, pivot_x, pivot_y) == above and all(np.linalg.norm([x - ex_x, y - ex_y]) > 4*self.cube_size for ex_x, ex_y, _ in existing_positions):
                    obj.set_pose(Pose([x, y, self.cube_size], euler2quat(0, 0, np.random.uniform(0, 2*np.pi))))
                    return x, y
            raise ValueError("Failed to find a valid position for the cube")


        # Generate position for the single cube on one specific side (e.g., above the line)
        try:
            single_cube_x, single_cube_y = generate_cube_position(True, self.env.unwrapped.objs[0], obstacle_initial_x, obstacle_initial_y, angle, existing_positions, self.cube_size)
        except ValueError as e:
            print(str(e))
            return False
        existing_positions.append([single_cube_x, single_cube_y,  self.env.unwrapped.objs[0].name])

        # Generate positions for the remaining cubes on the other side
        for obj in self.env.unwrapped.objs[1:]:
            cube_x, cube_y = generate_cube_position(False, obj, obstacle_initial_x, obstacle_initial_y, angle, existing_positions, self.cube_size)
            existing_positions.append([cube_x, cube_y, obj.name])
        self.existing_positions = existing_positions
        return True

    def a_star_with_avoidance(self, start, goal, existing_positions, scale, grid_size=200):
        """
        A* algorithm with obstacle avoidance.

        Args:
        start: Tuple (x, y) representing the start position.
        goal: Tuple (x, y) representing the goal position.
        existing_positions: A set of tuples (x, y) representing occupied positions.

        Returns:
        A list of tuples (x, y) representing the path, or None if no path is found.
        """

        def heuristic(a, b):
            # Manhattan distance as heuristic
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos):
            x, y = pos
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
                        (x+1, y+1), (x-1, y-1)]  # Adjacent cells
            # Filter out neighbors that are too close to existing positions or out of bounds
            return [(nx, ny) for nx, ny in neighbors if
                    0 <= nx < grid_size and 0 <= ny < grid_size and
                    all((nx - ex)**2 + (ny - ey)**2 > scale*2 for ex, ey in existing_positions)]

        # Initialize data structures
        open_set = set([start])
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        # A* search loop
        while open_set:
            # Node with lowest f_score
            current = min(open_set, key=lambda pos: f_score[pos])

            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1  # Assuming each step has cost 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        # No path found
        return None


    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path


    def get_grid_from_existing_positions(self, starting_cube_name, ending_cube_names, existing_positions, unit, scale, direction, grid_size=200):
        """
        starting_cube_name: str
        ending_cube_names: List[str]
        existing_positions: List[List[float, float, str]]
        unit: float
        direction: ["between", "left", "right", "front", "back"]
        """

        # convert to halfcube units
        positions_in_units = [(
            int(p[0] / unit + grid_size/2),
            int(p[1] / unit + grid_size/2),
            p[2]
        ) for p in existing_positions]

        grid = np.zeros((grid_size, grid_size))
        starting = None

        end = []
        for p in positions_in_units:
            if p[2] == starting_cube_name:
                starting = np.array(p[:2]).astype(int)
            if p[2] in ending_cube_names:
                end.append(np.array(p[:2]))

        starting = tuple(starting)
        if direction == "between":
            end = np.mean(end, axis=0).astype(int)
            end = tuple(end)
        else:
            max_dist = scale * 2 * np.sqrt(2)
            if direction == "right":
                end[0][1] -= max_dist
            elif direction == "left":
                end[0][1] += max_dist
            elif direction == "front":
                end[0][0] -= max_dist
            elif direction == "back":
                end[0][0] += max_dist
            end = tuple(end[0])

        for p in positions_in_units:
            # grid except starting point
            if p[2] == starting_cube_name:
                continue

            grid[p[0], p[1]] = 1
            if p[2] == "obstacle":
                # curr_scale = 2
                curr_scale = 2
            else:
                curr_scale = scale
                # curr_scale = 1
            for i in range(-curr_scale, curr_scale+1):
                for j in range(-curr_scale, curr_scale+1):
                    try:
                        x = (p[0] + i)
                        y = (p[1] + j)
                        grid[x, y] = 1
                    except IndexError:
                        pass
        return grid, starting, end



    def generate(self, N, random_trials = 5, data_dir=""):
        os.makedirs(data_dir, exist_ok=True)
        for n in range(N):
            env = self.create_env(self.obj_num, self.obstacle_num, self.scale)
            starting_cube_name = env.unwrapped.objs[0].name

            for t in range(random_trials):
                initialized = self.initialize_env()
                if not initialized: break
                directions = ["between", "left", "right", "front", "back"]
                direction = np.random.choice(directions)

                if direction == "between": selected_n = 2
                else: selected_n = 1

                ending_cube_names = np.random.choice([obj.name for obj in env.unwrapped.objs[1:]], selected_n, replace=False)

                grid_size = 200
                grid, start, goal = self.get_grid_from_existing_positions(starting_cube_name, ending_cube_names, self.existing_positions, self.unit, self.scale, direction, grid_size)

                a_star_positions = [(p[0], p[1]) for p in np.stack(np.where(grid == 1), 1)]

                path = self.a_star_with_avoidance(start, goal, a_star_positions, self.scale*4, grid_size)

                if path:
                    # print("Found path:", path)
                    path_grid = grid.copy()
                    for p in path:
                        path_grid[p[0], p[1]] = 2
                    # plt.imshow(path_grid[::-1, ::-1])

                    # convert to xy coordinates
                    xy_path = np.array([(np.array(p) - grid_size/2)*self.unit for p in path])

                    obj = [obj for obj in env.unwrapped.objs if obj.name == starting_cube_name][0]
                    initial_pose = obj.get_pose()
                    obj.set_pose(initial_pose)
                    count = 0

                    json_data = {
                        "imgs": [],
                        "initial_cube": obj.name,
                        "direction": direction,
                        "target_cubes": ending_cube_names.tolist(),
                    }
                    json_path = os.path.join(data_dir, f"data.json")

                    for i in range(xy_path.shape[0]):
                        obj.set_pose(Pose([xy_path[i][0], xy_path[i][1], self.cube_size], obj.get_pose().q))
                        if i % 9 == 0 or i == xy_path.shape[0] - 1:
                            # clear_output(wait=True)
                            obs, _, _, _, _ = env.step(env.action_space.sample())
                            img = obs["image"]["base_camera"]["rgb"]

                            # save frames
                            save_path = os.path.join(data_dir, f"avoid_collision_{n}_{t}_{count}.png")
                            Image.fromarray(img).save(save_path)
                            json_data["imgs"].append(save_path)
                            count += 1

                    try:
                        current_data = json.load(open(json_path, "r"))
                    except:
                        current_data = []
                    with open(json_path, "w") as f:
                        json.dump(current_data + [json_data], f)
                    print(f"Saved data for {n}-{t}")
                else:
                    print(f"No path found for {n}-{t}")