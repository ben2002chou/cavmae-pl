import json
import os

# Load JSON data
with open('/home/ben2002chou/code/cav-mae/data/audioset_eval_cleaned_2023.json', 'r') as file:
    data = json.load(file)

# List to hold entries to keep
filtered_data = []

# Directory to check for video files
video_directory = '/grand/EVITA/ben/AudioSet/eval/videos'

for entry in data['data']:
    video_id = entry['video_id']
    video_file_path = os.path.join(video_directory, video_id + '.mp4')
    
    # Check if the video file exists
    if os.path.exists(video_file_path):
        filtered_data.append(entry)

# Update the data key with filtered entries
data['data'] = filtered_data

# Write the filtered data back to a new JSON file
with open('/home/ben2002chou/code/cav-mae/data/audioset_eval_filtered.json', 'w') as file:
    json.dump(data, file, indent=4)
