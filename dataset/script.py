import os
import json
import shutil

# Directories
json_dir = './annots'
img_dir = './img'
aligned_img_dir = './img_aligned'
aligned_dir = './annots_aligned'

all_theta = []

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        with open(os.path.join(json_dir, filename), 'r') as file:
            im_info = json.load(file)
            all_theta.extend([detec['obb']['theta'] for detec in im_info['objects']])

print(all_theta)
min_theta = min(all_theta) if all_theta else None
max_theta = max(all_theta) if all_theta else None
print(f"Theta range: {min_theta} to {max_theta}")
            

# # Ensure output directories exist
# os.makedirs(aligned_img_dir, exist_ok=True)
# os.makedirs(aligned_dir, exist_ok=True)

# # Values of theta to check
# theta_values = {0, -0, 90, -90, 180, -180, 270, -270, 360, -360} # Use a set for faster lookup

# # Function to filter and save aligned annotations
# def filter_and_save_aligned(json_dir, aligned_dir, img_dir, aligned_img_dir, theta_values):
#     total_count = 0

#     for filename in os.listdir(json_dir):
#         if filename.endswith('.json'):
#             file_path = os.path.join(json_dir, filename)
#             with open(file_path, 'r') as file:
#                 data = json.load(file)

#             # Filter objects with desired theta values
#             aligned_objects = [
#                 obj for obj in data.get('objects', []) 
#                 if obj.get('obb', {}).get('theta') in theta_values
#             ]

#             if aligned_objects:  # Only process if there are matching objects
#                 # Save filtered JSON
#                 new_file_path = os.path.join(aligned_dir, filename)
#                 with open(new_file_path, 'w') as new_file:
#                     json.dump({"objects": aligned_objects}, new_file, indent=4)

#                 print(f"Saved {len(aligned_objects)} aligned objects in {new_file_path}")
#                 total_count += len(aligned_objects)

#                 # Move corresponding image if it exists
#                 img_filename = filename.replace('.json', '')  # Remove .json from filename
#                 img_path = os.path.join(img_dir, img_filename)
#                 if os.path.exists(img_path):  # Check if the image file exists
#                     shutil.copy(img_path, os.path.join(aligned_img_dir, img_filename))
#                     print(f"Copied image {img_filename} to {aligned_img_dir}")

#     print(f"Total aligned objects saved: {total_count}")

# # Run the function
# filter_and_save_aligned(json_dir, aligned_dir, img_dir, aligned_img_dir, theta_values)
