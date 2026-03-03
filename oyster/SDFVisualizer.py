import os
import time
import numpy as np
import matplotlib.pyplot as plt

from oyster.MapLoader import parse_xml_file, generate_map_from_cylinders
from py_planner import OccupancyGrid
from py_planner import MapLayer

def set_world(world_num):
    try:
        obstacles = []
        # offsets = [-5, 3]
        offsets = [-5 + 8 * i for i in range(len(world_num))]
        print(world_num)
        for i, world in enumerate(world_num):
            print(i)
            path = os.path.join(
                os.getenv("BARN_DATASET_PATH"),
                "world_files",
                f"world_{world}.world",
            )
            obs = parse_xml_file(path, offsets[i])
            if len(obstacles) == 0:
                obstacles = obs
            else:
                obstacles = np.concatenate((obstacles, obs))

    except Exception as e:
        obstacles = None
        print(
            "[ERROR] Loading obstacles did not work, has BARN dataset been installed and the BARN_DATASET_PATH",
            "environment variable been set to it's top level directory location?",
        )
        print(str(e))
        exit(0)

    occupancy_grid = generate_map_from_cylinders(obstacles, 0, 0, 0)
    map_util = OccupancyGrid(
        occupancy_grid.info.width,
        occupancy_grid.info.height,
        occupancy_grid.info.resolution,
        float(occupancy_grid.info.origin.position[0]),
        float(occupancy_grid.info.origin.position[1]),
        np.array(occupancy_grid.data),
        np.array([253, 254]),
        np.array([255]),
    )

    return map_util, occupancy_grid 

def main():

    map_util, grid = set_world([0])

    # plot
    w = grid.info.width
    h = grid.info.height
    res = grid.info.resolution
    ox = grid.info.origin.position[0]
    oy = grid.info.origin.position[1]

    data = np.array(grid.data).reshape((h, w))
    vis = np.where(data < 0, 0, data)

    # world coordinate bounds
    x_min = ox
    x_max = ox + w * res
    y_min = oy
    y_max = oy + h * res

    # draw onto main axis exactly once
    fig, ax = plt.subplots()
    grid_im = ax.imshow(
        vis,
        origin="lower",
        cmap="gray_r",
        extent=(x_min, x_max, y_min, y_max),
        interpolation="nearest",
        zorder=0,             # keep it behind everything else
    )


    x_coords = np.linspace(ox, (w-1) * res + ox, w)
    y_coords = np.linspace(oy, (h-1) * res + oy, h)

    sdf_matrix = np.zeros((h, w))

    first = False
    for i in range(h):
        for j in range(w):
            if x_coords[j] < -4 or x_coords[j] > -0.0:
                continue
            if y_coords[i] <-3.0 or y_coords[i] > 5:
                continue

            start = time.time()
            sdf_matrix[i, j] = map_util.sdf_dist(x_coords[j], y_coords[i], MapLayer.kInflated)
            end = time.time() - start

            if not first:
                print("sdf construction time:", end)
                first = True

    sdf_im = ax.imshow(
        sdf_matrix,
        origin="lower",
        cmap="viridis",
        extent=(x_min, x_max, y_min, y_max),
        alpha=0.4,
        zorder=1,
    )

    print(map_util.sdf_dist(-2.25, -2.5, MapLayer.kInflated))

    plt.colorbar(sdf_im, label="distance")
    plt.show()

if __name__ == "__main__":
    main()

