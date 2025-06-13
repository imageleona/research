#!/usr/bin/env bash

# Usage:
#   extract_frames.sh /path/to/video.mp4 /path/to/output_folder

# Capture arguments
INPUT_VIDEO="/mnt/c/Users/_s2111724/Documents/sequences/kizami_yuko_for_path_path.mp4"
OUTPUT_FOLDER="/mnt/c/Users/_s2111724/Documents/sequences/kizami_yuko_for_path_path_frames"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Run FFmpeg to extract frames
ffmpeg -i "$INPUT_VIDEO" "$OUTPUT_FOLDER/frame_%04d.png"

