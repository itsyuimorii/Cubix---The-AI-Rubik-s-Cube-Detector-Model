import os

# Define the directory containing your images
directory = "."

# Define the prefix and starting number
prefix = "Cube_"
image_number = 71

# Iterate over the files in the directory
for filename in enumerate(os.listdir(directory)):
    if not filename.endswith(".png"):
        continue
    # Generate the new filename
    new_filename = f"{prefix}{image_number:05}.png"
    
    # Rename the file
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    print(f"Renamed {filename} to {new_filename}")
    image_number += 1
