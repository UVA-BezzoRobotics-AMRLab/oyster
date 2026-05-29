import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Ensure we import the shared style and base class
from replay import BasePlotter, OCC_CMAP, FOLLOW_WINDOW

class BagJSONPlotter(BasePlotter):
    def __init__(self, json_path, fps=30, follow=False, load_obs=False, telemetry_only=False, save_path=None):
        # Load the pre-parsed JSON
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Map JSON structure to BasePlotter expectations
        self.meta_data      = self.data.get("metadata", {"model": "Bag Replay"})
        self.start          = self.data.get("start_pos", [0, 0, 0])
        self.goal           = self.data.get("goal_pos", [0, 0, 0])
        self.frames         = self.data.get("frames", [])
        self.fps            = fps
        self.follow         = follow
        self.telemetry_only = telemetry_only
        self.save_path      = Path(save_path) if save_path else None
        
        self.follow_window = 2

        # Initialize the figure from BasePlotter (handles layout switching natively)
        self._build_figure()
        
        # Internal map state
        self._map_img = None
        
        # Start animation or video saving workflow
        self.run()

    def update_map_visual(self, map_data):
        """
        Processes the 'map' key found within a frame.
        Handles the cropped dimensions and origin shift.
        Bypassed if telemetry_only is active.
        """
        if self.telemetry_only:
            return

        info = map_data["info"]
        w, h = info["width"], info["height"]
        res  = info["resolution"]
        ox   = info["origin"]["x"]
        oy   = info["origin"]["y"]

        # Convert the flat list back to a 2D grid for display
        raw = np.array(map_data["data"], dtype=np.int8).reshape((h, w))
        
        # Map values to our OCC_CMAP palette
        img = np.zeros((h, w), dtype=np.uint8)
        img[raw == 100] = 254  # Dark grey/black
        img[raw == -1]  = 253  # Transparent/Translucent
        
        # Calculate world-space boundaries
        extent = (ox, ox + w * res, oy, oy + h * res)

        if self._map_img is None:
            self._map_img = self.ax.imshow(
                img, origin="lower", cmap=OCC_CMAP,
                extent=extent, interpolation="nearest",
                vmin=0, vmax=255, zorder=0.1,
            )
        else:
            # Update data and shift the extent for the new crop
            self._map_img.set_data(img)
            self._map_img.set_extent(extent)

    def _update_frame(self, i):
        """
        Overrides BasePlotter's frame update to check for new map data.
        """
        frame = self.frames[i]
        
        # Check if this frame contains a map update (Only run if rendering spatial world)
        if "map" in frame and not self.telemetry_only:
            self.update_map_visual(frame["map"])
            
        # Call the parent update for robot, trajectory, and telemetry
        return super()._update_frame(i)

    def run(self):
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.frames),
            interval=1000 / self.fps,
            repeat=False,
        )
        
        if self.save_path:
            # Set up the writer. You can adjust bitrate if necessary for higher resolution grids.
            writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)
            print(f"Exporting animation to video file: {self.save_path}...")
            self.anim.save(self.save_path, writer=writer)
            print("Video export complete!")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Replay a Bag-to-JSON processed file.")
    parser.add_argument("json_file", help="Path to the processed JSON file")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--follow", action="store_true", help="Follow the robot")
    parser.add_argument("--telemetry-only", action="store_true", help="Turn off the 2D grid world plot")
    parser.add_argument("--save-to-video", type=str, default=None, metavar="OUTPUT.mp4",
                        help="Path where the recorded MP4 video should be saved")
    
    args = parser.parse_args()

    if not Path(args.json_file).exists():
        print(f"Error: File {args.json_file} not found.")
        return

    # Optional optimization: if writing directly to a file without displaying windows, 
    # you can swap to a non-interactive Matplotlib backend to save system overhead.
    if args.save_to_video:
        plt.switch_backend('Agg')

    BagJSONPlotter(
        json_path=args.json_file,
        fps=args.fps,
        follow=args.follow,
        telemetry_only=args.telemetry_only,
        save_path=args.save_to_video
    )

if __name__ == "__main__":
    main()
