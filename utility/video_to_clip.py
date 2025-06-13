import os
from moviepy.editor import VideoFileClip

def time_to_seconds(timestamp):
    """
    Convert a timestamp in the format 'minutes:seconds:milliseconds' to seconds.
    """
    minutes, seconds, milliseconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds + milliseconds / 1000

def extract_30_frame_clips(video_path, timestamps, total_frames=30, output_dir="C:/Users/_s2111724/Documents/shiai_test"):
    """
    Extracts clips with 20 frames around each timestamp and saves them as MP4 files.
    
    Parameters:
    - video_path: str, path to the video file
    - timestamps: list of strings in the format 'MM:SS:MS' where each clip will center
    - total_frames: int, total frames per clip (default is 20)
    - output_dir: str, directory to save the output clips
    
    Returns:
    - List of paths to the generated MP4 files
    """
    # Extract the base name of the video file (e.g., "video001" from "video001.mp4")
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the video
    video = VideoFileClip(video_path)
    frame_rate = video.fps  # Get the frame rate of the video
    clip_duration = total_frames / frame_rate  # Duration for 20 frames
    output_clips = []
    
    # Convert timestamps from 'MM:SS:MS' format to seconds
    timestamps_in_seconds = [time_to_seconds(timestamp) for timestamp in timestamps]
    
    for i, center_time in enumerate(timestamps_in_seconds):
        # Calculate the start and end times for the clip
        start_time = max(0, center_time - (clip_duration / 2))
        end_time = min(video.duration, center_time + (clip_duration / 2))
        
        # Extract the clip
        clip = video.subclip(start_time, end_time)
        
        # Set the output path and save the clip as an MP4 file
        output_path = os.path.join(output_dir, f"{base_filename}_clip_{i+1:03d}.mp4")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        output_clips.append(output_path)
    
    video.close()
    return output_clips

# Example usage
video_path = "C:/Users/_s2111724/Documents/old/karate-clip/clip1-3.mp4"  # Replace with your video file path
timestamps = ["00:03:681"]  # Example timestamps
output_clips = extract_30_frame_clips(video_path, timestamps)
print("Clips generated:", output_clips)
