import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from rosbags.rosbag1 import Reader as Reader1
    from rosbags.rosbag2 import Reader as Reader2
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    sys.exit("Please install rosbags: pip install rosbags")

def _get_yaw(q):
    return float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))

def crop_occupancy_grid(msg):
    w, h = msg.info.width, msg.info.height
    data = np.array(msg.data, dtype=np.int8).reshape((h, w))
    
    known_indices = np.where(data != -1)
    if len(known_indices[0]) == 0:
        return None

    rmin, rmax = np.min(known_indices[0]), np.max(known_indices[0])
    cmin, cmax = np.min(known_indices[1]), np.max(known_indices[1])

    cropped_data = data[rmin:rmax+1, cmin:cmax+1]
    res = msg.info.resolution
    new_ox = msg.info.origin.position.x + (cmin * res)
    new_oy = msg.info.origin.position.y + (rmin * res)

    return {
        "info": {
            "width": int(cropped_data.shape[1]),
            "height": int(cropped_data.shape[0]),
            "resolution": float(res),
            "origin": {"x": float(new_ox), "y": float(new_oy)}
        },
        "data": cropped_data.flatten().tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="Resampled ROS Bag to JSON Exporter")
    parser.add_argument("bag", help="Path to ROS1 bag or ROS2 folder")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--fps", type=float, default=15.0, help="Target replay frequency")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    out_path = Path(args.output or bag_path.with_suffix(".json"))
    
    fmt = "ros1" if bag_path.suffix == ".bag" else "ros2"
    Reader = Reader1 if fmt == "ros1" else Reader2
    typestore = get_typestore(Stores.ROS1_NOETIC if fmt == "ros1" else Stores.ROS2_HUMBLE)

    # Added the two alpha tracking channels to the topic map
    topic_map = {
        "/gmapping/odometry": "odom",
        "/cmd_vel": "vel",
        "/tube_viz": "tubes",
        "/MINCO_path": "traj",
        "/mpc_prediction": "mpc",
        "/map": "map",
        "/cbf_alpha_abv": "alpha_upper",
        "/cbf_alpha_blw": "alpha_lower"
    }

    # Internal state buffers
    latest_state = {
        "robot_pos": [0.0, 0.0, 0.0],
        "velocity": 0.0,
        "alpha_upper": 0.0,
        "alpha_lower": 0.0,
        "mpc_horizon": [],
        "trajectory": {"knots": [], "xs": [], "ys": []},
        "tubes": {"top": [], "bottom": []}
    }
    current_map_json = None
    map_updated_in_window = False
    frames = []

    print(f"[*] Reading {bag_path.name}...")
    
    with Reader(bag_path) as reader:
        start_ts = reader.start_time
        end_ts = reader.end_time
        interval_ns = int(1e9 / args.fps)
        
        conns = [c for c in reader.connections if c.topic in topic_map]
        msg_generator = reader.messages(connections=conns)
        
        try:
            current_msg = next(msg_generator)
            
            for target_ts in tqdm(range(start_ts, end_ts, interval_ns), desc="Resampling"):
                
                while current_msg and current_msg[1] <= target_ts:
                    conn, ts, raw = current_msg
                    
                    raw_bytes = raw if fmt == "ros1" else bytes(raw)
                    msg = typestore.deserialize_ros1(raw_bytes, conn.msgtype) if fmt == "ros1" else typestore.deserialize_cdr(raw_bytes, conn.msgtype)
                    
                    topic = conn.topic
                    if topic == "/gmapping/odometry":
                        p = msg.pose.pose.position
                        latest_state["robot_pos"] = [p.x, p.y, _get_yaw(msg.pose.pose.orientation)]
                        # latest_state["velocity"] = float(msg.twist.twist.linear.x)
                        # latest_state["velocity"] = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
                    elif topic =="/cmd_vel": 
                        vel = np.linalg.norm([msg.linear.x, msg.linear.y])

                        if latest_state["velocity"] <= 0.1 or vel >= 1e-6:
                            latest_state["velocity"] = np.linalg.norm([msg.linear.x, msg.linear.y])
                    
                    elif topic == "/cbf_alpha_abv":
                        latest_state["alpha_upper"] = float(msg.data)

                    elif topic == "/cbf_alpha_blw":
                        latest_state["alpha_lower"] = float(msg.data)
                    
                    elif topic == "/tube_viz":
                        for i, marker in enumerate(msg.markers):
                            pts = [[p.x, p.y] for p in marker.points]
                            if i == 0: latest_state["tubes"]["top"] = pts
                            elif i == 1: latest_state["tubes"]["bottom"] = pts
                    
                    elif topic == "/MINCO_path":
                        latest_state["trajectory"] = {
                            "knots": list(range(len(msg.points))),
                            "xs": [p.x for p in msg.points], "ys": [p.y for p in msg.points]
                        }

                    elif topic == "/mpc_prediction":
                        poses = msg.poses if hasattr(msg, 'poses') else []
                        latest_state["mpc_horizon"] = [[p.pose.position.x, p.pose.position.y] for p in poses]

                    elif topic == "/map":
                        cropped = crop_occupancy_grid(msg)
                        if cropped:
                            current_map_json = cropped
                            map_updated_in_window = True

                    try:
                        current_msg = next(msg_generator)
                    except StopIteration:
                        current_msg = None

                # Create a deep-copy snapshot for this time slice
                snap = {
                    "robot_pos": list(latest_state["robot_pos"]),
                    "velocity": latest_state["velocity"],
                    "alpha_upper": latest_state["alpha_upper"],
                    "alpha_lower": latest_state["alpha_lower"],
                    "mpc_horizon": [list(p) for p in latest_state["mpc_horizon"]],
                    "trajectory": {
                        "knots": list(latest_state["trajectory"]["knots"]),
                        "xs": list(latest_state["trajectory"]["xs"]),
                        "ys": list(latest_state["trajectory"]["ys"])
                    },
                    "tubes": {
                        "top": [list(p) for p in latest_state["tubes"]["top"]],
                        "bottom": [list(p) for p in latest_state["tubes"]["bottom"]]
                    }
                }
                
                if map_updated_in_window:
                    snap["map"] = current_map_json
                    map_updated_in_window = False
                
                if len(snap["tubes"]["top"]) > 0 and len(snap["tubes"]["bottom"]) > 0:
                    frames.append(snap)

        except StopIteration:
            pass

    if not frames:
        sys.exit("No frames generated. Safety tube arrays were entirely empty throughout tracking.")

    output_data = {
        "metadata": {
            "model": "ROSBAG_RESAMPLED",
            "max_speed": max([f["velocity"] for f in frames]),
            "fps": args.fps,
            "total_frames": len(frames)
        },
        "start_pos": frames[0]["robot_pos"],
        "goal_pos": frames[-1]["robot_pos"],
        "frames": frames
    }

    print(f"[*] Saving {len(frames)} sanitized frames to {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(output_data, f)

    print(f"[!] Success!")

if __name__ == "__main__":
    main()
