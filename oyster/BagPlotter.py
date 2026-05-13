"""
replay_from_bag.py
==================
Feeds a ROS1 (.bag) or ROS2 (.db3 / directory) rosbag into the MPCC live
plotter.  All figure/drawing logic lives in replay_plotter.BasePlotter;
this file only adds map ingestion and bag reading.

Usage
-----
    python replay_from_bag.py recording.bag \
        --frame-topic /mpcc/frame \
        --map-topic   /map \
        --start-pos 0 0 --goal-pos 10 5

    # Play back as fast as possible, keep robot centred:
    python replay_from_bag.py recording.bag \
        --frame-topic /odom --map-topic /map \
        --no-realtime --follow
"""

import argparse
import json
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── rosbags (pip install rosbags) ─────────────────────────────────────────────
try:
    from rosbags.rosbag1 import Reader as Reader1
    HAS_ROS1 = True
except ImportError:
    HAS_ROS1 = False

try:
    from rosbags.rosbag2 import Reader as Reader2
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from rosbags.typesys import get_typestore, Stores

# ── All plotting logic lives here ─────────────────────────────────────────────
from replay_plotter import BasePlotter, OCC_CMAP


# ─────────────────────────────────────────────────────────────────────────────
# BagReplayPlotter — BasePlotter + live OccupancyGrid ingestion
# ─────────────────────────────────────────────────────────────────────────────

class BagReplayPlotter(BasePlotter):
    """
    Extends BasePlotter with update_map(), which accepts a deserialized
    nav_msgs/OccupancyGrid and renders it into the main axes.  No world files,
    no cylinder generation — the map comes entirely from the bag.
    """

    def __init__(self, start_pos, goal_pos, meta_data=None, follow=False):
        self.start     = start_pos
        self.goal      = goal_pos
        self.meta_data = meta_data
        self.follow    = follow
        self.frames    = []     # accumulated via push_frame()
        self._build_figure()    # inherited from BasePlotter
        plt.ion()
        plt.show(block=False)

    def update_map(self, grid_msg):
        """
        Push a nav_msgs/OccupancyGrid into the axes.  Safe to call on every
        map message (latched or live-updating).

        OccupancyGrid convention:
            -1  → unknown  → rendered as near-transparent grey  (index 253)
             0  → free     → transparent                        (index 0)
            100 → occupied → dark                               (index 254)
        """
        info = grid_msg.info
        w, h = info.width, info.height
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y

        raw = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))
        img = np.zeros((h, w), dtype=np.uint8)
        img[raw == 100] = 254
        img[raw == -1]  = 253

        extent = (ox, ox + w * res, oy, oy + h * res)

        if self._map_img is None:
            self._map_img = self.ax.imshow(
                img, origin="lower", cmap=OCC_CMAP,
                extent=extent, interpolation="nearest",
                vmin=0, vmax=255, zorder=0.1,
            )
        else:
            self._map_img.set_data(img)
            self._map_img.set_extent(extent)

        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Message → frame dict  (extend for your own message types)
# ─────────────────────────────────────────────────────────────────────────────

def _msg_to_frame(msg, typestr: str) -> dict | None:
    """
    Convert a rosbags message to a BasePlotter frame dict.  Return None to skip.

    Recognised keys (all optional except robot_pos):
        robot_pos    [x, y, theta]
        velocity     float
        alpha_upper  float
        alpha_lower  float
        cbf_upper    float
        cbf_lower    float
        mpc_horizon  [[x, y], ...]
        trajectory   {knots, xs, ys}
        tubes        {top: [[x,y],...], bottom: [[x,y],...]}
    """

    # Easiest integration: publish a std_msgs/String whose .data is the
    # JSON-encoded frame dict (same schema as the existing JSON replay files).
    if "std_msgs/msg/String" in typestr or "std_msgs/String" in typestr:
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError:
            return None

    # nav_msgs/Odometry
    if "Odometry" in typestr:
        p   = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y),
                               1 - 2*(q.y*q.y + q.z*q.z)))
        return {"robot_pos": [p.x, p.y, yaw],
                "velocity":  float(msg.twist.twist.linear.x)}

    # geometry_msgs/PoseStamped
    if "PoseStamped" in typestr:
        p   = msg.pose.position
        q   = msg.pose.orientation
        yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y),
                               1 - 2*(q.y*q.y + q.z*q.z)))
        return {"robot_pos": [p.x, p.y, yaw]}

    # nav_msgs/Path → treat as MPC horizon; first pose = robot position
    if "Path" in typestr:
        if not msg.poses:
            return None
        p0 = msg.poses[0].pose.position
        return {
            "robot_pos":   [p0.x, p0.y, 0.0],
            "mpc_horizon": [[p.pose.position.x, p.pose.position.y]
                            for p in msg.poses],
        }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Bag player
# ─────────────────────────────────────────────────────────────────────────────

class BagPlayer:
    def __init__(
        self,
        bag_path:    str,
        frame_topic: str,
        map_topic:   str,
        start_pos:   list[float],
        goal_pos:    list[float],
        fps:         float = 30.0,
        realtime:    bool  = True,
        follow:      bool  = False,
        meta_data:   dict  = None,
    ):
        self.bag_path    = Path(bag_path)
        self.frame_topic = frame_topic
        self.map_topic   = map_topic
        self.start_pos   = start_pos
        self.goal_pos    = goal_pos
        self.fps         = fps
        self.realtime    = realtime
        self.follow      = follow
        self.meta_data   = meta_data
        self._detect_format()

    def _detect_format(self):
        if self.bag_path.suffix == ".bag":
            if not HAS_ROS1:
                sys.exit("ROS1 bag support unavailable: pip install rosbags")
            self.fmt = "ros1"
        else:
            if not HAS_ROS2:
                sys.exit("ROS2 bag support unavailable: pip install rosbags")
            self.fmt = "ros2"

    def _open_reader(self):
        return (Reader1 if self.fmt == "ros1" else Reader2)(self.bag_path)

    def _typestore(self):
        return get_typestore(
            Stores.ROS1_NOETIC if self.fmt == "ros1" else Stores.ROS2_HUMBLE
        )

    def run(self):
        typestore = self._typestore()

        plotter = BagReplayPlotter(
            start_pos = self.start_pos,
            goal_pos  = self.goal_pos,
            meta_data = self.meta_data,
            follow    = self.follow,
        )

        topics_wanted   = {self.frame_topic, self.map_topic}
        last_wall       = time.monotonic()
        last_bag_ts     = None

        with self._open_reader() as reader:
            connections = [c for c in reader.connections
                           if c.topic in topics_wanted]

            for conn, bag_ts, rawdata in reader.messages(connections=connections):
                if plotter._closed:
                    break

                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

                # Realtime pacing
                if self.realtime and last_bag_ts is not None:
                    bag_dt = (bag_ts - last_bag_ts) * 1e-9
                    if (sleep := bag_dt - (time.monotonic() - last_wall)) > 0:
                        time.sleep(sleep)
                last_bag_ts = bag_ts
                last_wall   = time.monotonic()

                if conn.topic == self.map_topic:
                    plotter.update_map(msg)
                elif conn.topic == self.frame_topic:
                    frame = _msg_to_frame(msg, conn.msgtype)
                    if frame is not None:
                        plotter.push_frame(frame)

        print("Bag finished — close the window to exit.")
        plt.ioff()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a ROS1/ROS2 bag in the MPCC live plotter."
    )
    parser.add_argument("bag",            help=".bag file or ROS2 recording directory")
    parser.add_argument("--frame-topic",  required=True,
                        help="Topic carrying per-frame robot data")
    parser.add_argument("--map-topic",    default="/map",
                        help="nav_msgs/OccupancyGrid topic (default: /map)")
    parser.add_argument("--start-pos",    nargs=2, type=float, default=[0.0, 0.0],
                        metavar=("X", "Y"))
    parser.add_argument("--goal-pos",     nargs=2, type=float, default=[10.0, 0.0],
                        metavar=("X", "Y"))
    parser.add_argument("--fps",          type=float, default=30.0)
    parser.add_argument("--no-realtime",  action="store_true",
                        help="Play back as fast as possible")
    parser.add_argument("--follow",       action="store_true",
                        help="Keep robot centred in the overhead view")
    parser.add_argument("--model",        default=None)
    parser.add_argument("--max-speed",    type=float, default=None)
    args = parser.parse_args()

    BagPlayer(
        bag_path    = args.bag,
        frame_topic = args.frame_topic,
        map_topic   = args.map_topic,
        start_pos   = args.start_pos,
        goal_pos    = args.goal_pos,
        fps         = args.fps,
        realtime    = not args.no_realtime,
        follow      = args.follow,
        meta_data   = ({"model": args.model, "max_speed": args.max_speed or "?"}
                       if args.model else None),
    ).run()


if __name__ == "__main__":
    main()
