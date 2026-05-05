import click
import numpy as np

from datetime import datetime
from oyster.CBFEnv import CBFEnv
from oyster.Logger import Logger
from oyster.RobotMPC import Dynamics


@click.command()
@click.option("--world_num", default=0)
@click.option("--task_num", default=0)
def main(world_num, task_num):
    env = CBFEnv(world_num=[world_num], normalize_obs=False)

    titles = {
        6: "di_1_5",
        7: "di_2_5",
        8: "uni_1_8",
        9: "uni_1_5",
    }

    failed = False
    while not failed:
        env.reset_task(task_num)

        now = datetime.now()
        date_str = now.strftime("%Y_%m_%d_%H%M%S")
        logger = Logger(f"world_{env.world_nums[0]}_{titles[task_num]}_{date_str}.json")

        logger.log_static_obstacles(env.obstacles)
        logger.log_start_and_goal(env._start, env._goal)
        logger.log_meta_data(env.params)

        done = False
        while not done:
            obs, reward, done, _ = env.step([0, 0])
            if env.dynamic_model == Dynamics.UNICYCLE:
                velocity = env.robot_state[3]
            else:
                velocity = np.linalg.norm(env.robot_state[3:5])

            horizon = env.mpc.get_horizon()
            mpc_horizon = np.column_stack((horizon.states.xs[:], horizon.states.ys[:]))

            tube_upper_pts, tube_lower_pts = env._compute_tube_pts()
            logger.log_frame(
                env.robot_state[:3],
                velocity,
                obs,
                mpc_horizon,
                (env.knots, env.xs, env.ys),
                tube_upper_pts,
                tube_lower_pts,
            )

            env.render()
            if env.closest_obstacle_pass < 0.075:
                failed = True
                logger.save()


if __name__ == "__main__":
    main()
