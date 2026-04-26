"""Script to play a checkpoint and save attention weights as videos for each robot."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play an RL agent and save attention videos for each robot."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_start_step", type=int, default=0, help="Start step for the simulation."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--debug", action="store_true", default=False, help="Enable debug mode."
)
parser.add_argument(
    "--no_resume",
    default=None,
    action="store_true",
    help="Force play in no resume mode.",
)
# custom play arguments
parser.add_argument(
    "--env_cfg",
    action="store_true",
    default=False,
    help="Load configuration from file.",
)
parser.add_argument(
    "--agent_cfg",
    action="store_true",
    default=False,
    help="Load configuration from file.",
)
parser.add_argument(
    "--sample",
    action="store_true",
    default=False,
    help="Sample actions instead of using the policy.",
)
parser.add_argument(
    "--zero_act_until", type=int, default=0, help="Zero actions until this timestep."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./attention_videos",
    help="Directory to save attention videos.",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="FPS for the output videos.",
)
parser.add_argument(
    "--head_idx",
    type=int,
    default=None,
    help="Which attention head to visualize (None = average all heads).",
)
parser.add_argument(
    "--colormap",
    type=str,
    default="hot",
    choices=["hot", "jet", "viridis", "plasma"],
    help="Colormap for attention heatmap.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Blending factor for attention overlay on depth image (0-1).",
)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def get_colormap(name: str):
    """Get OpenCV colormap by name."""
    colormaps = {
        "hot": cv2.COLORMAP_HOT,
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    return colormaps.get(name, cv2.COLORMAP_HOT)


def attention_to_heatmap(
    attn_map: np.ndarray,
    colormap: int,
    target_size: tuple | None = None,
    depth_image: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Convert attention weights to a colored heatmap image.

    Args:
        attn_map: Attention weights array (H, W) or (num_heads, H, W)
        colormap: OpenCV colormap constant
        target_size: Optional (width, height) to resize the output
        depth_image: Optional depth image (H, W) to overlay attention on
        alpha: Blending factor for attention overlay (0-1)

    Returns:
        RGB heatmap image as uint8 array
    """
    # Normalize attention to 0-255
    attn_min = attn_map.min()
    attn_max = attn_map.max()
    if attn_max - attn_min < 1e-8:
        attn_normalized = np.zeros_like(attn_map)
    else:
        attn_normalized = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)

    # Convert to uint8
    attn_uint8 = (attn_normalized * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(attn_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # If depth image is provided, overlay attention on depth
    if depth_image is not None:
        # Normalize depth image to 0-255
        depth_min = depth_image.min()
        depth_max = depth_image.max()
        if depth_max - depth_min < 1e-8:
            depth_normalized = np.zeros_like(depth_image)
        else:
            depth_normalized = (depth_image - depth_min) / (
                depth_max - depth_min + 1e-8
            )
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Convert depth to RGB
        depth_rgb = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)

        # Resize attention heatmap to match depth if needed
        if heatmap.shape[:2] != depth_rgb.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (depth_rgb.shape[1], depth_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Blend depth and attention
        heatmap = cv2.addWeighted(depth_rgb, 1 - alpha, heatmap, alpha, 0)

    # Resize if target size specified
    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    return heatmap


def main():
    """Play with Instinct-RL agent and save attention videos."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(
        args_cli.task, args_cli
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run),
                os.path.basename(agent_cfg.load_run),
                agent_cfg.load_checkpoint,
            )
        else:
            resume_path = get_checkpoint_path(
                log_root_path, args_cli.load_run, agent_cfg.load_checkpoint
            )
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(
            f"[INFO] No experiment directory specified. Using default: {log_root_path}"
        )
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    env_cfg_path = os.path.join(log_dir, "params", "env.pkl")
    agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")

    if args_cli.env_cfg:
        if os.path.exists(env_cfg_path):
            import pickle

            with open(env_cfg_path, "rb") as f:
                env_cfg = pickle.load(f)
        else:
            print(
                f"[WARNING] env.pkl not found at {env_cfg_path}. Using default environment configuration."
            )
    elif agent_cfg.load_run is not None and os.path.exists(env_cfg_path):
        import pickle

        with open(env_cfg_path, "rb") as f:
            env_cfg = pickle.load(f)

    if args_cli.agent_cfg:
        if os.path.exists(agent_cfg_path):
            agent_cfg_dict = load_yaml(agent_cfg_path)
        else:
            print(
                f"[WARNING] agent.yaml not found at {agent_cfg_path}. Using default agent configuration."
            )
            agent_cfg_dict = agent_cfg.to_dict()  # type: ignore
    elif agent_cfg.load_run is not None and os.path.exists(agent_cfg_path):
        agent_cfg_dict = load_yaml(agent_cfg_path)
    else:
        agent_cfg_dict = agent_cfg.to_dict()  # type: ignore

    # create isaac environment (no rendering needed for attention visualization)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
    )
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # setup output directory
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Attention videos will be saved to: {output_dir}")

    # get number of environments
    num_envs = env.num_envs
    print(f"[INFO] Number of environments (robots): {num_envs}")
    print(f"[INFO] Recording {args_cli.video_length} steps of attention")

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Initialize video writers for each robot
    # We'll create them after we know the attention map size
    video_writers = {}
    colormap = get_colormap(args_cli.colormap)

    # Storage for frames per robot
    robot_frames = {i: [] for i in range(num_envs)}

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # simulate environment
    print("[INFO] Starting simulation...")
    while simulation_app.is_running() and timestep < args_cli.video_length:
        # run everything in inference mode
        with torch.inference_mode():
            # Get attention weights during forward pass
            encoder_output = ppo_runner.alg.actor_critic.encoders(
                obs, return_attn_weights=True
            )

            if isinstance(encoder_output, tuple) and len(encoder_output) == 2:
                _, attn_dict = encoder_output
                if attn_dict and "depth_attention" in attn_dict:
                    attn = attn_dict[
                        "depth_attention"
                    ]  # Shape: (num_envs, num_heads, H, W)

                    # Process attention for visualization
                    attn_np = attn.cpu().numpy()  # (num_envs, num_heads, H, W)

                    # Extract depth image from observation
                    # Get observation segments to find depth_image
                    obs_segments = env.get_obs_segments()
                    from instinct_rl.utils.utils import (
                        get_obs_slice,
                        get_subobs_by_components,
                    )

                    depth_slice, depth_shape = get_obs_slice(
                        obs_segments, "depth_image"
                    )
                    depth_obs = obs[:, depth_slice].cpu().numpy()
                    # Reshape to (num_envs, num_frames, H, W)
                    depth_obs = depth_obs.reshape(num_envs, *depth_shape)
                    # Take the last frame for visualization
                    depth_frames = depth_obs[:, -1]  # (num_envs, H, W)

                    # Select which head(s) to visualize
                    if args_cli.head_idx is not None:
                        # Visualize specific head
                        if args_cli.head_idx < attn_np.shape[1]:
                            attn_viz = attn_np[:, args_cli.head_idx]  # (num_envs, H, W)
                        else:
                            print(
                                f"[WARNING] head_idx {args_cli.head_idx} out of range, using average"
                            )
                            attn_viz = attn_np.mean(axis=1)  # (num_envs, H, W)
                    else:
                        # Average across all heads
                        attn_viz = attn_np.mean(axis=1)  # (num_envs, H, W)

                    # Convert each robot's attention to heatmap overlaid on depth
                    for env_id in range(num_envs):
                        heatmap = attention_to_heatmap(
                            attn_viz[env_id],
                            colormap,
                            target_size=(320, 180),  # Resize for better visibility
                            depth_image=depth_frames[env_id],
                            alpha=args_cli.alpha,
                        )
                        robot_frames[env_id].append(heatmap)

            # Get actions from policy
            actions = policy(obs)

            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0

            # env stepping
            obs, rewards, dones, infos = env.step(actions)

        timestep += 1

        if timestep % 100 == 0:
            print(f"[INFO] Processed {timestep}/{args_cli.video_length} steps")

    # Close the simulator
    env.close()

    print(f"[INFO] Simulation complete. Saving {num_envs} videos...")

    # Save videos for each robot
    for env_id in range(num_envs):
        if len(robot_frames[env_id]) == 0:
            print(f"[WARNING] No frames captured for robot {env_id}, skipping...")
            continue

        # Get video dimensions from first frame
        frame_height, frame_width = robot_frames[env_id][0].shape[:2]

        # Setup video writer
        video_path = os.path.join(output_dir, f"robot_{env_id:03d}_attention.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        video_writer = cv2.VideoWriter(
            video_path, fourcc, args_cli.fps, (frame_width, frame_height)
        )

        # Write all frames
        for frame in robot_frames[env_id]:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(
            f"[INFO] Saved video for robot {env_id}: {video_path} ({len(robot_frames[env_id])} frames)"
        )

    print(f"[INFO] All videos saved to: {output_dir}")

    # Open output directory
    subprocess.run(["xdg-open", output_dir], capture_output=True)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
