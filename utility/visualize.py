import json
import cv2
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for proper rendering
import matplotlib.pyplot as plt
import numpy as np

def visualize_keypoints_to_video(json_file, output_video_path, video_size=(1600, 1600), fps=15):
    """
    Creates a video visualizing keypoints on an XY plane from the first sample of the JSON dataset.

    Args:
        json_file (str): Path to the JSON file containing keypoint data.
        output_video_path (str): Path to save the output video.
        video_size (tuple): Size of the video frames (width, height).
        fps (int): Frames per second for the output video.
    """
    # Load the JSON dataset
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)

    
    for i in range(100):
        sample = data["data"][i]
        total_frames = 0

        for frame_number, frame_keypoints in enumerate(sample, start=1):
        # Create a blank image for plotting
            fig, ax = plt.subplots(figsize=(8, 8))

            # Flatten and group the keypoints as (x, y) pairs
            keypoints = frame_keypoints  # Each frame has its keypoints
            xy_coords = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)]

            # Plot the keypoints
            for x, y in xy_coords:
                ax.scatter(x, 1600-1*y, c='red', s=40)
                ax.text(x, -1*y, f"({x:.1f}, {y:.1f})", fontsize=8)

            # Customize plot
            ax.set_title(f"Frame {frame_number}")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.invert_yaxis()  # Invert Y-axis for image-like coordinate system
            ax.grid(True)
            plt.xlim(0, video_size[0])
            plt.ylim(0, video_size[1])

            # Save the plot as an image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, 1:]  # Convert ARGB to RGB by dropping alpha channel

            # Convert RGB (matplotlib) to BGR (OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Write frame to video
            video_writer.write(cv2.resize(img, video_size))
            total_frames += 1

            # Close the plot to save memory
            plt.close(fig)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")
    print(f"Total frames processed: {total_frames}")

# Example Usage
json_file = "C:/Users/_s2111724/training/data5_skeleton/data5_skeleton_training/k0002_yuko/k0002_yuko_test.json"  # Path to your JSON dataset
output_video_path = "C:/Users/_s2111724/Documents/sotsuron/yuko_test_visualize.mp4"  # Output video path
visualize_keypoints_to_video(json_file, output_video_path)