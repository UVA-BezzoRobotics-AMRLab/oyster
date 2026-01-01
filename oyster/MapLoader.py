import os
import math
import argparse
import numpy as np
import xml.dom.minidom

from lxml import etree

class Pose:

    def __init__(self):
        self.position = np.array([])
        self.orientation = np.array([])

class OccInfo:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.resolution = 0
        self.origin = Pose()

class OccupancyGrid:

    def __init__(self):
        self.info = OccInfo()
        self.data = []


def parse_xml_file(xml_file):
    # parse the xml file

    cylinder_data = []

    tree = etree.parse(xml_file)
    root = tree.getroot()

    all_models = root.findall(".//model")

    unit_cylinder_models = [
        model
        for model in all_models
        if model.get("name").startswith("unit_cylinder_")
        and model.find("static") is not None
    ]

    for model in unit_cylinder_models:
        pose_element = model.find("./pose")
        pose = pose_element.text.strip().split()
        position = [float(p) for p in pose[:3]]
        radius = 0.075

        # shift everything back 5m in y direction
        cylinder_data.append([position[0], position[1] - 5.0, radius])

    return cylinder_data

def generate_map_from_cylinders(cylinders, theta, dx, dy,
                                obstacle_inflation=0.30,   # meters
                                inflation_cost=253,
                                obstacle_cost=254):
    grid = OccupancyGrid()

    grid.info.width = 600
    grid.info.height = 600
    grid.info.resolution = 0.05
    grid.info.origin = Pose()
    grid.info.origin.position = np.array([-15., -15., 0.])
    grid.info.origin.orientation = np.array([0, 0, 0, 1])

    grid.data = [0] * grid.info.width * grid.info.height

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    for cylinder in cylinders:
        x, y, radius = cylinder

        x, y = np.matmul(R, np.array([x, y])) + [dx, dy]

        grid_x_center = int((x - grid.info.origin.position[0]) / grid.info.resolution)
        grid_y_center = int((y - grid.info.origin.position[1]) / grid.info.resolution)

        # Convert radii to grid units
        r_cells = int(radius / grid.info.resolution)
        r_inf_cells = int((radius + obstacle_inflation) / grid.info.resolution)

        for gx in range(grid_x_center - r_inf_cells, grid_x_center + r_inf_cells + 1):
            for gy in range(grid_y_center - r_inf_cells, grid_y_center + r_inf_cells + 1):

                if not (0 <= gx < grid.info.width and 0 <= gy < grid.info.height):
                    continue

                dist = math.sqrt((gx - grid_x_center)**2 + (gy - grid_y_center)**2) * grid.info.resolution

                idx = gy * grid.info.width + gx

                # Inner obstacle
                if dist <= radius:
                    grid.data[idx] = obstacle_cost

                elif dist <= radius + obstacle_inflation:
                    if grid.data[idx] != obstacle_cost:
                        grid.data[idx] = inflation_cost

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("world_file", help="The name of the file to process.")

    args = parser.parse_args()

    world_file = args.world_file

    rot = 0
    dx = 0
    dy = 0

    occupancy_grid = OccupancyGrid()

    # Replace with your XML file path
    if world_file.endswith(".world"):

        if not os.path.exists(world_file):
            raise ValueError("Please provide a world file path")

        obstacles = parse_xml_file(world_file)

        # Generate occupancy grid message from obstacles
        occupancy_grid = generate_map_from_cylinders(obstacles, rot, dx, dy)

    else:
        raise ValueError("Invalid arguments")
        exit(-1)


if __name__ == "__main__":
    main()
