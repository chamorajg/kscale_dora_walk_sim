"""Trains a humanoid to stand up."""

import argparse
import sys
sys.path.append("./")
import isaacgym
import torch
from sim.envs import task_registry
from sim.utils.helpers import get_args
from tdmpc.src import logger
from tdmpc.src.algorithm.helper import Episode, ReplayBuffer
from tdmpc.src.algorithm.tdmpc import TDMPC
from dataclasses import dataclass, field
from datetime import datetime
from isaacgym import gymapi
from typing import List
import time
import numpy as np
from pathlib import Path
import random
torch.backends.cudnn.benchmark = True
__LOGS__ = "logs"

@dataclass
class TDMPC_DoraConfigs:
	seed: int = 42
	task : str = "walk"
	exp_name : str = "dora"
	device : str = "cuda:0"
	num_envs : int = 10

	lr : float = 5e-4
	modality : str = "state"
	enc_dim: int = 256
	mlp_dim = [512, 512]
	latent_dim: int = 50
	
	iterations : int = 6
	num_samples : int = 512
	num_elites : int = 50
	mixture_coef : float = 0.05
	min_std : float = 0.05
	temperature : float = 0.5
	momentum : float = 0.1
	horizon : int = 5
	std_schedule: str = f"linear(0.5, {min_std}, 50000)"
	horizon_schedule: str = f"linear(1, {horizon}, 25000)"

	batch_size: int = 1024
	max_buffer_size : int = 1000000
	reward_coef : float = 1
	value_coef : float = 0.5
	consistency_coef : float = 2
	rho : float = 0.5
	kappa : float = 0.1
	per_alpha: float = 0.6
	per_beta : float =  0.4
	grad_clip_norm : float =  10
	seed_steps: int = 3000
	update_freq: int = 2
	tau: int = 0.01

	discount : float = 0.99
	buffer_device : str = "cpu"
	train_steps : int = int(1e6)
	num_q : int = 5

	action_repeat : int = 4
	eval_freq: int = 15000
	eval_episodes : int = 1

	save_model : bool = True
	save_video : bool = False

	use_wandb  : bool = False
	wandb_entity : str = "crajagopalan"
	wandb_project : str = "xbot"
	


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(test_env, agent, h1, num_episodes, step, env_step, video, action_repeat=1):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, privileged_obs = test_env.reset()
		critic_obs = privileged_obs if privileged_obs is not None else obs	
		state = torch.cat([obs, critic_obs], dim=-1)[0] if privileged_obs is not None else obs[0]
		dones, ep_reward, t = torch.tensor([False]), 0, 0
		if video: video.init(test_env, h1, enabled=(i==0))
		while not dones[0].item():
			actions = agent.plan(state, eval_mode=True, step=step, t0=t==0)
			for _ in range(action_repeat):
				obs, privileged_obs, rewards, dones, infos = test_env.step(actions)
				critic_obs = privileged_obs if privileged_obs is not None else obs
				state = torch.cat([obs, critic_obs], dim=-1)[0] if privileged_obs is not None else obs[0]
				ep_reward += rewards[0]
				if video: video.record(test_env, h1)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return torch.nanmean(torch.tensor(episode_rewards)).item()

def train(args: argparse.Namespace) -> None:
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()	
	env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
	env, _ = task_registry.make_env(name=args.task, args=args)	

	tdmpc_cfg = TDMPC_DoraConfigs

	if tdmpc_cfg.save_video:
		env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

		camera_properties = gymapi.CameraProperties()
		camera_properties.width = 160
		camera_properties.height = 120
		h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
		camera_offset = gymapi.Vec3(3, -3, 1)
		camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
		actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
		body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
		env.gym.attach_camera_to_body(
			h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
		)

	set_seed(tdmpc_cfg.seed)
	now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	work_dir = Path().cwd() / __LOGS__ / f"{now}_{tdmpc_cfg.task}_{tdmpc_cfg.modality}_{tdmpc_cfg.exp_name}"

	obs, privileged_obs = env.reset()
	critic_obs = privileged_obs if privileged_obs is not None else obs	
	state = torch.cat([obs, critic_obs], dim=-1)[0] if privileged_obs is not None else obs[0]
	
	tdmpc_cfg.obs_shape = [state.shape[0]]
	tdmpc_cfg.action_shape = (env.num_actions)
	tdmpc_cfg.action_dim = env.num_actions
	episode_length = 60
	tdmpc_cfg.episode_length = episode_length # int(env.max_episode_length // tdmpc_cfg.action_repeat)
	tdmpc_cfg.num_envs = env.num_envs
	
	L = logger.Logger(work_dir, tdmpc_cfg)	
	
	agent = TDMPC(tdmpc_cfg)
	buffer = ReplayBuffer(tdmpc_cfg)
	fp = None # "/home/guest/sim/logs/walk_state_dora_42/models/tdmpc_policy_33.pt"
	# agent.load(fp)
	init_step = 0
	episode_idx, start_time = 0, time.time()
	if fp is not None:
		episode_idx = int(fp.split(".")[0].split("_")[-1])
		init_step = episode_idx * int(env.max_episode_length // tdmpc_cfg.action_repeat)
	for step in range(init_step, tdmpc_cfg.train_steps + int(env.max_episode_length // tdmpc_cfg.action_repeat), int(env.max_episode_length // tdmpc_cfg.action_repeat)):
		obs, privileged_obs = env.reset()
		critic_obs = privileged_obs if privileged_obs is not None else obs	
		state = torch.cat([obs, critic_obs], dim=-1) if privileged_obs is not None else obs
		episode = Episode(tdmpc_cfg, state)
		for i in range(episode_length): # int(env.max_episode_length // tdmpc_cfg.action_repeat)):
			actions = agent.plan(state, t0 = i == 0, eval_mode=False, step=step)
			original_state = state.clone()
			total_rewards = []
			total_dones = []
			for _ in range(tdmpc_cfg.action_repeat):
				obs, privileged_obs, rewards, dones, infos = env.step(actions)
				critic_obs = privileged_obs if privileged_obs is not None else obs
				state = torch.cat([obs, critic_obs], dim=-1) if privileged_obs is not None else obs
				total_rewards.append(rewards)
				total_dones.append(dones)
			episode += (original_state, actions, torch.stack(total_rewards).sum(dim=0), torch.stack(total_dones).any(dim=0), torch.stack(total_dones).any(dim=0))
		assert len(episode) == episode_length # int(env.max_episode_length // tdmpc_cfg.action_repeat)
		buffer += episode
		
		# Update model
		train_metrics = {}
		if step >= tdmpc_cfg.seed_steps:
			num_updates = 10 # tdmpc_cfg.seed_steps if step == tdmpc_cfg.seed_steps else int(env.max_episode_length // tdmpc_cfg.action_repeat)
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step) # * tdmpc_cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward.sum().item() / env.num_envs}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')
		L.save(agent, f"tdmpc_policy_{int(step // int(env.max_episode_length // tdmpc_cfg.action_repeat))}.pt")
		# # Evaluate agent periodically
		# if env_step % tdmpc_cfg.eval_freq == 0:
		# 	common_metrics['episode_reward'] = evaluate(env, agent, h1 if L.video is not None else None, tdmpc_cfg.eval_episodes, step, env_step, L.video, tdmpc_cfg.action_repeat)
		# 	L.log(common_metrics, category='eval')

	
	print('Training completed successfully')

if __name__ == "__main__":
    # python -m sim.humanoid_gym.train
    train(get_args())