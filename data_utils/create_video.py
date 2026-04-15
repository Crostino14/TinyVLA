import h5py
import numpy as np
import cv2
import os
import argparse

def hdf5_to_video(hdf5_path, output_dir=None, fps=20):
    """
    Converts the 'camera_front_image' frames from an HDF5 file into an MP4 video.

    Args:
        hdf5_path (str): Path to the .hdf5 file.
        output_dir (str): Directory where the video will be saved.
        fps (int): Frames per second for the video.
    """
    # Open the HDF5 file
    print(f"Opening HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        # Check available datasets
        if "/observations/images/camera_front_image" not in f:
            raise KeyError("Dataset '/observations/images/camera_front_image' not found in HDF5 file.")
        
        # Load the dataset into memory (or stream if large)
        images = f["/observations/images/camera_front_image"]
        num_frames, height, width, channels = images.shape
        print(f"Loaded {num_frames} frames with shape ({height}, {width}, {channels})")

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(hdf5_path)
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(hdf5_path))[0] + "_front.mp4"
        video_path = os.path.join(output_dir, video_name)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write each frame
        for i in range(num_frames):
            frame = images[i]
            # Ensure correct dtype
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Convert RGB → BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"✅ Video saved at: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 camera_front_image dataset to video.")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the .hdf5 file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the video")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second of the output video")
    args = parser.parse_args()

    hdf5_to_video(args.hdf5_path, args.output_dir, args.fps)
