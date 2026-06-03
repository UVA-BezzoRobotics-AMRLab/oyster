import os
import click
import numpy as np
from oyster.CBFEnv import CBFEnv


def load_world_list(file_path):
    """Reads world numbers from a file, assuming one per line."""
    if not os.path.exists(file_path):
        print(f"Error: World list file {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


@click.command()
@click.option(
    "--world_list_file", required=True, help="Path to file containing world numbers."
)
def main(world_list_file):
    # 1. Schedule of alpha values
    # alpha_schedule = [-1, 0.5, 3.0, 4.0, 6.0, 8.0]
    alpha_schedule = [-1, 3.0, 8.0]

    # 2. Load worlds from supplied file
    world_nums = load_world_list(world_list_file)
    if not world_nums:
        return

    # Initialize environment
    env = CBFEnv()
    tasks = env.get_all_task_idx()
    # As per your snippet: testing the last 4 tasks
    eval_tasks = [tasks[6], tasks[7]]

    # 3. Sweep iterate for every alpha value
    for alpha in alpha_schedule:
        # 4. Create new file for this specific alpha
        outfile = f"cbf_bicycle_{alpha}.txt"

        # Initialize the new file with headers
        with open(outfile, "w", newline="") as f:
            f.write("Below are the simulation results for the test trials\n")
            f.write("WORLD\tTASK\tSUCCESS\tSTEPS\tLOWEST_CLEARANCE\n")

        print(f"Starting sweep for alpha: {alpha}")

        for world_n in world_nums:
            env.set_world([world_n])

            for task in eval_tasks:
                # Based on your snippet: running 5 trials per world/task combo
                for i in range(5):
                    env.reset_task(task)
                    env.params.cbf.alpha_abv = alpha
                    env.params.cbf.alpha_blw = alpha

                    if alpha < 0:
                        env.params.cbf.alpha_abv = 10
                        env.params.cbf.alpha_blw = 10
                        env.params.cbf.use_cbf = False
                        env.set_mpc(env.params)

                    done = False
                    while not done:
                        # Taking step with [0,0] as per your logging snippet logic
                        obs, reward, done, _ = env.step([0, 0])

                        # Optional: env.render() if you want to watch the sweep
                        # env.render()

                    # Logging using the requested format
                    with open(outfile, "a", newline="") as f:
                        f.write(
                            "%d\t%d\t%d\t%d\t%.3f\n"
                            % (
                                world_n,
                                task,
                                not env.did_collide,
                                env.step_count,
                                env.closest_obstacle_pass,
                            )
                        )

        print(f"Completed results saved to {outfile}")


if __name__ == "__main__":
    main()
