import os

folder_path = r"C:\Users\User\Documents\RMIT\Year 2\Data Comm\Assignment 3\raw\diversity"
prefix = "diversity-inclusion_"

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    # Skip directories
    if os.path.isdir(old_path):
        continue

    new_filename = prefix + filename
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)

print("✅ Finished renaming files.")
