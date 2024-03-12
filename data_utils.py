import h5py
from mani_skill2.utils.io_utils import load_json

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def get_nth_episode_info(dataset_file, n):
    """
        Get the nth episode from the dataset
        Arguments:
            - dataset_file: ex: "ManiSkill2/demos/v0/rigid_body/PushChair-v1/3020/trajectory.h5"
            - n: episode number to load
        Returns
            - env_info: contains environmnt informatin
            - env_state: contains environment state to load
            - actions: contains actions taken in the environment
    """

    data = h5py.File(dataset_file, "r")
    json_path = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]
    env_info = json_data["env_info"]
    env_info["env_kwargs"]["obs_mode"] = 'rgbd'

    eps = episodes[n]
    trajectory = data[f"traj_{eps['episode_id']}"]
    trajectory = load_h5_data(trajectory)
    
    env_states = trajectory["env_states"]
    init_env_state = env_states[0]
    final_env_state = env_states[-1]

    actions = trajectory['actions']
    reset_kwargs = eps["reset_kwargs"]
    seed = reset_kwargs.get("seed", None)

    return env_info, init_env_state, final_env_state, actions, reset_kwargs, seed
