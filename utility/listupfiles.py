import os
import pandas as pd
import cv2

def windows_to_unix_path(win_path):
    """
    Converts a Windows file path to a Unix-style path for WSL compatibility.
    """
    return win_path.replace("\\", "/").replace("C:/", "/mnt/c/")

def list_files_to_excel(folder_path, output_excel, fps, prefix, video_extensions=None):
    """
    Reads filenames inside a folder, calculates frame numbers and durations, and writes them into an Excel file.

    Args:
        folder_path (str): The path to the folder containing the files.
        output_excel (str): The output Excel file path.
        fps (float): Frames per second for the video files to calculate frame durations.
        prefix (str): The prefix to prepend to each filename in the output.
        video_extensions (list): List of file extensions to consider as video files (default: common formats).
    """
    try:
        # Set default video extensions if none are provided
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        # Check if FPS is valid
        if fps <= 0:
            print("Error: FPS must be greater than 0.")
            return

        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder path '{folder_path}' does not exist.")
            return

        # Get a list of all files in the folder
        file_list = os.listdir(folder_path)
        if not file_list:
            print(f"Error: No files found in folder '{folder_path}'.")
            return

        # Sort the file list
        file_list.sort()
        print(f"Found {len(file_list)} files in the folder.")

        # Initialize lists for DataFrame columns
        frame_numbers = []
        durations = []  # Duration in seconds
        full_file_paths = []

        for file in file_list:
            video_path = os.path.join(folder_path, file)

            # Prepend the prefix to the file path and convert to Unix style
            prefixed_path = os.path.join(prefix, os.path.relpath(video_path, folder_path)).replace("\\", "/")
            unix_path = windows_to_unix_path(prefixed_path)
            full_file_paths.append(unix_path)

            # Check if the file has a valid video extension
            if not video_path.lower().endswith(tuple(video_extensions)):
                print(f"Skipping non-video file: {file}")
                frame_numbers.append(None)
                durations.append(None)
                continue

            # Open the video file and get frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video file: {file}")
                frame_numbers.append(None)
                durations.append(None)
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_numbers.append(frame_count)

            # Calculate video duration (in seconds)
            duration = frame_count / fps if fps > 0 else None
            durations.append(duration)

            print(f"Processed video: {file}, Frame Count: {frame_count}, Duration: {duration:.2f} seconds")
            cap.release()

        # Create a DataFrame
        df = pd.DataFrame({
            "Filename": full_file_paths,  # Use Unix-style paths
            "Index": "",  # Leave blank for manual entry
            "Frame Number": frame_numbers,
            "Duration (seconds)": durations  # Add duration column
        })

        # Write the DataFrame to an Excel file
        df.to_excel(output_excel, index=False, sheet_name="File List")

        print(f"File list has been written to '{output_excel}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the folder path, output Excel file, FPS, and prefix
folder_path = "C:/Users/_s2111724/detectron2-code/video-data-by-moves/kizami"  # Replace with your folder path
output_excel = "C:/Users/_s2111724/detectron2-code/video-data-by-moves/file_list_kizami.xlsx"  # Output Excel file
fps = 60.0  # Replace with the actual FPS of your video files
#prefix = "/home/appuser/detectron2-code/video_data/yuko/karate-clip-yuko"  # Prefix to prepend to each filename
prefix = "/home/appuser/detectron2-code/video-data-by-moves/kizami"
# Call the function
list_files_to_excel(folder_path, output_excel, fps, prefix)
