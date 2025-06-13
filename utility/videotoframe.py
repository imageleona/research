import os
import subprocess
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image


def time_to_seconds(timestamp):
    """
    Convert a timestamp in the format 'minutes:seconds:milliseconds' to seconds.
    """
    minutes, seconds, milliseconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds + milliseconds / 1000


def extract_and_save_frames(video_path, timestamps, frame_duration=0.033, total_frames=20, output_folder="output_frames"):
    """
    Extracts frames around given timestamps and saves them as JPEG images.
    """
    video = VideoFileClip(video_path)
    timestamps_in_seconds = [time_to_seconds(ts) for ts in timestamps]

    all_frames = []
    for i, center_time in enumerate(timestamps_in_seconds):
        start_time = max(0, center_time - (total_frames // 2) * frame_duration)
        frame_times = [start_time + j * frame_duration for j in range(total_frames)]

        timestamp_folder = os.path.join(output_folder, f"timestamp_{i+1}")
        os.makedirs(timestamp_folder, exist_ok=True)

        for j, frame_time in enumerate(frame_times):
            if frame_time > video.duration:
                break  # Skip frames beyond the video duration
            frame = video.get_frame(frame_time)
            frame_path = os.path.join(timestamp_folder, f"frame_{j+1:03d}.jpg")
            Image.fromarray(frame).save(frame_path)
            all_frames.append(frame_path)
    
    video.close()
    return all_frames


# Main logic to process videos from Excel file
def process_videos_with_frames(excel_file_path, timestamps, output_base_path):
    """
    Processes videos listed in an Excel file, extracting frames around timestamps and saving them.
    """
    # Read the Excel file
    df = pd.read_excel(excel_file_path)

    for _, row in df.iterrows():
        video_path = row["Filename"]  # Assuming the column is named "Filename"
        video_filename = os.path.basename(video_path)
        folder_name = os.path.splitext(video_filename)[0]  # Folder name without extension
        output_folder = os.path.join(output_base_path, folder_name)

        # Create output folder for JPEGs
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing video: {video_path}")
        
        # Extract frames around timestamps and save as JPEG
        extract_and_save_frames(video_path, timestamps, output_folder=output_folder)

        # Optionally log success
        print(f"Frames extracted and saved for {video_filename}")


# Parameters and file paths
file_path = "/mnt/c/Users/_s2111724/Documents/karate-clip2/file_list.xlsx"
output_base_path = "/mnt/c/Users/_s2111724/Documents/karate_clip_sam2_jpeg"
timestamps = ["00:07:500", "00:12:200", "00:16:400", "00:21:100"]  # Example timestamps

# Run the processing
process_videos_with_frames(file_path, timestamps, output_base_path)
