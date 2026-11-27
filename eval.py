"""
eval.py

Author: Caleb Scott

---

Evaluation of trained models.
For the purpose of seeing the frame + focus.

"""

# IMPORTS
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import torch
from ruamel import yaml
from torchvision.utils import make_grid
from tqdm import tqdm

from configs.constants import LUMA_VALS
from configs.globals import ROM
from modules import environment as e
from modules import utils
from modules.actor_critic import ActorCriticPolicy
from modules.agent import Agent
from modules.wm import WorldModel


# CONSTANTS
DEFAULT_PATCH_COLOR = np.array([255, 255, 0])

# MAIN
def rgb_to_gray_3ch(frames: np.ndarray) -> np.ndarray:
    """Converts an RGB uint8 array to 3-channel grayscale."""
    weights = np.array(LUMA_VALS)
    gray = np.sum(frames * weights, axis=-1, keepdims=True)
    gray_3ch = np.repeat(gray, 3, axis=-1)
    return gray_3ch.astype(frames.dtype)

def paint_square(
        frame: np.ndarray, 
        center: np.ndarray, 
        rgb: np.ndarray, 
        r: np.ndarray,
        hollow: bool = False
    ) -> None:
    """Paints a square dot centered at (y, x) in-place, clipped to frame boundaries."""
    h, w = frame.shape[:2]
    y, x = center[0], center[1]
    ry, rx = r[0], r[1]

    y_top = y - ry
    y_bot = y + ry
    x_left = x - rx
    x_right = x + rx

    y_min = max(0, y_top)
    y_max = min(h, y_bot + 1)
    x_min = max(0, x_left)
    x_max = min(w, x_right + 1)

    # Draw filled/empty box
    if not hollow:
        frame[y_min:y_max, x_min:x_max] = rgb
    else:
        # Stop if the box is completely outside the frame
        if x_min >= x_max or y_min >= y_max:
            return

        # Paint each edge only if that edge lies inside the frame
        if 0 <= y_top < h:
            frame[y_top, x_min:x_max] = rgb
        if 0 <= y_bot < h:
            frame[y_bot, x_min:x_max] = rgb
        if 0 <= x_left < w:
            frame[y_min:y_max, x_left] = rgb
        if 0 <= x_right < w:
            frame[y_min:y_max, x_right] = rgb

def paint_frames(frames: list, patches: list, res: list, height: int, width: int, color: np.ndarray = DEFAULT_PATCH_COLOR):
    # Paint the focus on each frame
    for recon, patch in zip(frames, patches):
        # Paint center "focus"
        r = np.array([2,2])
        paint_square(recon, patch, color, r)

        # Paint multi-resolution, if applicable
        for resolution in res:
            h, w = height * resolution, width * resolution
            paint_square(recon, patch, color, np.array([h, w]), hollow=True)

def viz_frames_video(frame_data: dict, res: list, patch_height: int, patch_width: int, src_dir: str, dest_dir: str):
    """ Given frame data, compile into video with painted focus """

    # Paint the focus points on the frame
    frames = [rgb_to_gray_3ch(f) for f in frame_data['frames']]
    patches = frame_data['patch_pos']
    assert len(frames) == len(patches)
    paint_frames(frames, patches, res, patch_height, patch_width)

    # Save the frames as mp4 file.
    src = src_dir.replace("/", ".")
    dest = Path(dest_dir + src + "sacc-track.mp4")

    with iio.imopen(dest, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=30)
        for frame in tqdm(frames, desc="Encoding video", unit="frame"):
            writer.write_frame(frame)

    print(f"Wrote {len(frames)} frames to {dest}...")

def viz_frames_eval_sample(frame_data: dict, num_recons: int, res: list, patch_width: int, patch_height: int, rng):
    """ Given frame data from running the eye wrapper, collect recons. """

    # Sample random sequence of frames.
    num_frames = len(frame_data['frames'])
    frames_idx = rng.choice(num_frames - num_recons)
    recons_start, recons_end = frames_idx, frames_idx + num_recons
    print(f"Number of recons: {num_recons}")
    print(f"Number of frames: {num_frames}")
    print(f"Selected range: {(recons_start, recons_end)}")

    # Slice frames and convert to grayscale.
    recons_frames = frame_data['frames'][recons_start: recons_end]
    recons_frames = [rgb_to_gray_3ch(r.copy()) for r in recons_frames]
    patches = frame_data['patch_pos'][recons_start: recons_end]

    # Paint the frames
    paint_frames(recons_frames, patches, res, patch_height, patch_width)

    # Convert to tensor, then make grid
    frames_tensor = torch.from_numpy(np.stack(recons_frames)).permute(0, 3, 1, 2)

    grid_tensor = make_grid(frames_tensor, nrow=4, padding=2)
    grid_image = grid_tensor.permute(1, 2, 0).numpy()

    return grid_image

def main(game=ROM.Pong_NoFrameskip):
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help='Device used for training')
    parser.add_argument("--game", type=str, default=game, help="Choice of Atari game")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--compile", default=False, action='store_true', help="Whether to use torch.compile")
    parser.add_argument("--dir", type=str, default="models/pong/sacc/", help="Base dir for models.")
    parser.add_argument("--num", type=int, default=1, help="Default number of episodes.")
    parser.add_argument("--recons", type=int, default=8, help="Default number of episodes.")
    parser.add_argument("--no_viz", default=True, action='store_false', help="Flag for disabling visualizing sampled frames.")
    parser.add_argument("--record", default=False, action='store_true', help="Flag for recording eval.")
    parser.add_argument("--record_dir", type=str, default="recordings/", help="Default dir for recordings.")
    args = parser.parse_args()

    # Load from config file
    with open(args.config, 'r') as f:
        config = yaml.YAML(typ='safe', pure=True).load(f)

    # Update config with command line args
    config = {
        **config, 
        'config': args.config, 
        'game': args.game,
        'seed': args.seed,
        'compile': args.compile,
        'dir': args.dir,
        'num': args.num,
        'recons': args.recons,
        'no_viz': args.no_viz,
        'record': args.record,
        'record_dir': args.record_dir
    }
    config = SimpleNamespace(**config)
    # print(f"Config: {config}")

    # Device, autocast, compile
    device = torch.device(args.device)
    compile_ = lambda mod: torch.compile(mod, dynamic=False, disable=not config.compile)

    # Env, policy, agent, wm
    seed = (config.seed + 42) * 27
    rng = utils.seed_all(seed)
    
    # Create eye-sacchade game wrapper env
    print("Creating env...")
    env_selection = e.eye_env
    env = env_selection(config.game, **config.env)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}\n")

    y_dim = config.wm['y_dim']
    a_dim = env.action_space.n

    print("Making Agent/WM...")
    policy = ActorCriticPolicy(
        y_dim,
        a_dim,
        config.policy['actor'],
        config.policy['critic'],
        compile_=compile_,
        device=device
    )
    agent = Agent(policy, env.action_space, config.action_stack)
    wm = WorldModel(
        env.observation_space, 
        agent.stack_act_space, 
        **config.wm, 
        compile_=compile_, 
        device=device
    )

    print("Importing WM weights...")
    wm_path = Path(config.dir + "wm.pt")
    wm_checkpoint = torch.load(wm_path, map_location=device, weights_only=True) 
    clean_wm_checkpoint = {
        key.replace("_orig_mod.", ""): value
        for key, value in wm_checkpoint.items()
    }
    wm.load_state_dict(clean_wm_checkpoint, strict=True)

    print("Importing Agent weights...")
    ag_path = Path(config.dir + "agent.pt")
    ag_checkpoint = torch.load(ag_path, map_location=device, weights_only=True) 
    clean_ag_checkpoint = {
        key.replace("_orig_mod.", ""): value
        for key, value in ag_checkpoint.items()
    }
    agent.load_state_dict(clean_ag_checkpoint, strict=True)

    sacc_multi = False
    if "multi" in wm_path:
        print("multi-res sacc detected...")
        sacc_multi = True

    print("Putting into eval mode and running...\n")

    wm.eval()
    agent.eval()

    stats = []
    frame_data = []

    with torch.no_grad():
        for i in range(config.num):
            # Observation
            o, _ = env.reset()
            done = False
            rew = 0.0
            data = []

            encode = wm.encode
            policy = agent.policy

            while not done:
                o = torch.tensor(o, dtype=torch.uint8, device=device).unsqueeze(0)
                y = encode(o)
                a = policy(y)

                # Observation, reward, termination, truncation, info
                o, reward, term, trunc, info = env.step(a.item())
                data.append(info)
                done = term or trunc
                rew += reward

            print(f"Episode {i} final reward: {rew}")
            stats.append({
                "episode": i,
                "total reward": rew
            })

            frame_data.append({
                "episode": i,
                "multi_res": sacc_multi,
                "frames": [item['frame'] for item in data],
                "patch_pos": [item['patch_pos'] for item in data],
            })

    h, w = config.env['patch_size'], config.env['patch_size']
    res = [1.0]
    if frame_data[0]['multi_res']:
        res.append(2.0)

    if not config.no_viz:
        print("Sampled Frame Visualization: \n")
        grid = viz_frames_eval_sample(frame_data[0], config.recons, res, h, w, rng)
        plt.figure(figsize=(8, 9))
        plt.imshow(grid)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if config.record:
        viz_frames_video(frame_data[0], res, h, w, config.dir, config.record_dir)

if __name__ == "__main__":
    main()