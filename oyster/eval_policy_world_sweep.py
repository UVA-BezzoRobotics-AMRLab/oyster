import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from matplotlib import pyplot as plt
from oyster.rlkit.torch.sac.policies import TanhGaussianPolicy
from oyster.rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from oyster.rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_training import deep_update_dict
from oyster.rlkit.torch.sac.policies import MakeDeterministic
from oyster.rlkit.samplers.util import rollout
from oyster.CBFEnv import CBFEnv


def load_world_list(file_path):
    """Reads world numbers from a file, assuming one per line."""
    if not os.path.exists(file_path):
        print(f"Error: World list file {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def sim_policy(
    variant,
    path_to_exp,
    world_num=0,
    start_task_idx=0,  # New argument
    num_trajs=1,
    deterministic=False,
    save_video=False,
    manual_step=False,
    world_nums=[],
):
    max_path_length = 250
    env = CBFEnv(
        world_num=[world_num],
        manual_step=manual_step,
        save_video=save_video,
        max_step_count=max_path_length,
        N=3,
    )

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # Filter tasks based on the starting task index provided
    all_eval_tasks = list(tasks[-variant["n_eval_tasks"] :])
    eval_tasks = [t for t in all_eval_tasks if t >= start_task_idx]

    print(
        "Testing starting from Task {} ({} tasks total), starting from World {}".format(
            start_task_idx, len(eval_tasks), world_num
        )
    )

    # ... [Network instantiation code remains the same] ...
    latent_dim = variant["latent_size"]
    context_encoder_output_dim = (
        latent_dim * 2
        if variant["algo_params"]["use_information_bottleneck"]
        else latent_dim
    )
    reward_dim = 1
    net_size = variant["net_size"]
    recurrent = variant["algo_params"]["recurrent"]
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=(
            2 * obs_dim + action_dim + reward_dim
            if variant["algo_params"]["use_next_obs_in_context"]
            else obs_dim + action_dim + reward_dim
        ),
        output_size=context_encoder_output_dim,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(latent_dim, context_encoder, policy, **variant["algo_params"])
    if deterministic:
        agent = MakeDeterministic(agent)

    itr = 74
    cpu_device = None if torch.cuda.is_available() else torch.device("cpu")
    context_encoder.load_state_dict(
        torch.load(
            os.path.join(path_to_exp, f"context_encoder_itr_{itr}.pth"),
            map_location=cpu_device,
        )
    )
    policy.load_state_dict(
        torch.load(
            os.path.join(path_to_exp, f"policy_itr_{itr}.pth"), map_location=cpu_device
        )
    )

    outfile = "cbf_adapt_gcopter.txt"
    # Only write header if we are starting from the beginning
    if world_num == 0 and (not eval_tasks or eval_tasks[0] == all_eval_tasks[0]):
        with open(outfile, "w", newline="") as f:
            f.write("Below are the simulation results for the test trials\n")
            f.write("WORLD\tTASK\tSUCCESS\tSTEPS\tLOWEST_CLEARANCE\n")

    if world_nums == []:
        world_nums = list(range(world_num, 300))

    # Loop through tasks and worlds
    for idx in eval_tasks:
        # Start world_n from world_num, then increment to 300
        for world_n in world_nums:
            env.set_world([world_n])
            env.reset_task(idx)
            agent.clear_z()
            updated = False

            for n in range(5):
                path = rollout(
                    env,
                    agent,
                    max_path_length=max_path_length,
                    accum_context=True,
                    save_frames=False,
                    animated=False,
                )

                with open(outfile, "a", newline="") as f:
                    f.write(
                        "%d\t%d\t%d\t%d\t%.3f\n"
                        % (
                            world_n,
                            idx,
                            not env.did_collide,
                            env.step_count,
                            env.closest_obstacle_pass,
                        )
                    )

                if not updated and n >= variant["algo_params"]["num_exp_traj_eval"]:
                    updated = True
                    agent.infer_posterior(agent.context)

        # After completing all worlds for the first task,
        # reset world_num to 0 so the NEXT task starts from World 0
        world_num = 0


@click.command()
@click.argument("config", default=None)
@click.argument("path", default=None)
@click.option("--world_num", default=0, type=int, help="Starting world number")
@click.option("--task_num", default=0, type=int, help="Starting task index")
@click.option("--num_trajs", default=3)
@click.option("--stochastic", is_flag=True, default=False)
@click.option("--video", is_flag=True, default=False)
@click.option("--manual_step", is_flag=True, default=False)
@click.option(
    "--world_list_file", default="", help="Path to file containing world numbers."
)
def main(
    config,
    path,
    world_num,
    task_num,
    num_trajs,
    stochastic,
    video,
    manual_step,
    world_list_file,
):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    world_nums = load_world_list(world_list_file)

    sim_policy(
        variant,
        path,
        world_num,
        task_num,
        num_trajs,
        not stochastic,
        video,
        manual_step,
        world_nums,
    )


if __name__ == "__main__":
    main()
