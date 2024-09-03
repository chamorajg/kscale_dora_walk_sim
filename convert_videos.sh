#!/bin/bash

# Check if a directory is provided as an argument, otherwise use the current directory
if [ -z "$1" ]; then
    directory="."
else
    directory="$1"
fi

# Ensure the provided directory exists
if [ ! -d "$directory" ]; then
    echo "Directory $directory does not exist."
    exit 1
fi

# Loop through all video files in the specified directory
for file in "$directory"/*; do
    # Check if it's a file and has a video file extension
    if [ -f "$file" ] && [[ "$file" =~ \.(mp4|mkv|avi|mov|flv|wmv)$ ]]; then
        # Get the base name and extension
        base_name="${file%.*}"
        extension="${file##*.}"
        
        # Set the output file name with "_changed" appended before the extension
        output_file="${base_name}_changed.${extension}"
        
        # Convert the video to 25 fps
        ffmpeg -i "$file" -vf "fps=25" "$output_file"
        
        # Notify user of successful conversion
        echo "Converted $file to $output_file"
    fi
done
