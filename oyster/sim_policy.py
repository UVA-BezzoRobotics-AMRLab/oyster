import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from RobotEnv import RobotEnv
from matplotlib import pyplot as plt
from oyster.rlkit.torch.sac.policies import TanhGaussianPolicy
from oyster.rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from oyster.rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_training import deep_update_dict
from oyster.rlkit.torch.sac.policies import MakeDeterministic
from oyster.rlkit.samplers.util import rollout
from oyster.CBFEnv import CBFEnv


def sim_policy(
    variant, path_to_exp, world_num=0, num_trajs=1, deterministic=False, save_video=False, manual_step=False
):
    """
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg

    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    """

    # create multi-task environment and sample tasks
    max_path_length = 500
    env = CBFEnv(world_num=[world_num, 280], manual_step=manual_step, save_video=save_video, max_step_count=max_path_length)
    # env = RobotEnv(randomize_traj=True)
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    # eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    N = 1
    # eval_tasks=list(tasks[-variant['n_eval_tasks']+ N -1:-variant['n_eval_tasks'] + N])
    eval_tasks=list([tasks[-1]])
    print(
        "testing on {} test tasks, {} trajectories each".format(
            len(eval_tasks), num_trajs
        )
    )

    # instantiate networks
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
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    itr = 68
    cpu_device = None if torch.cuda.is_available() else torch.device('cpu')
    print(cpu_device)
    context_encoder.load_state_dict(
        torch.load(os.path.join(path_to_exp, f"context_encoder_itr_{itr}.pth"), map_location=cpu_device)
    )
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, f"policy_itr_{itr}.pth"), map_location=cpu_device))

    # loop through tasks collecting rollouts
    all_rets = []
    # video_frames = []
    total_rets = []
    for idx in eval_tasks:
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        for n in range(num_trajs):
            path = rollout(
                env,
                agent,
                max_path_length=max_path_length,
                accum_context=True,
                save_frames=False,
                animated=True,
            )
            paths.append(path)
            # if save_video:
            #     video_frames += [t['frame'] for t in path['env_infos']]
            if n >= variant["algo_params"]["num_exp_traj_eval"]:
                agent.infer_posterior(agent.context)

            total_rets.append([env.total_reward, env.is_success])

        all_rets.append([sum(p["rewards"]) for p in paths])

        if save_video:
            from datetime import datetime

            current_datetime = datetime.now()
            date_str = current_datetime.strftime("%H_%M_%S_%d-%m-%Y")
            plt.savefig(f"./output/{date_str}")

    # if save_video:
    #     # save frames to file temporarily
    #     temp_dir = os.path.join(path_to_exp, 'temp')
    #     os.makedirs(temp_dir, exist_ok=True)
    #     for i, frm in enumerate(video_frames):
    #         frm.save(os.path.join(temp_dir, '%06d.jpg' % i))
    #
    #     video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
    #     # run ffmpeg to make the video
    #     os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
    #     # delete the frames
    #     shutil.rmtree(temp_dir)

    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=0)
    for i, ret in enumerate(rets):
        print("trajectory {}, avg return: {} \n".format(i, ret))

    print("total rets:")
    print(total_rets)


@click.command()
@click.argument("config", default=None)
@click.argument("path", default=None)
@click.option("--world_num", default=0)
@click.option("--num_trajs", default=3)
@click.option("--deterministic", is_flag=True, default=False)
@click.option("--video", is_flag=True, default=False)
@click.option("--manual_step", is_flag=True, default=False)
def main(config, path, world_num, num_trajs, deterministic, video, manual_step):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, world_num, num_trajs, deterministic, video, manual_step)


if __name__ == "__main__":
    main()
