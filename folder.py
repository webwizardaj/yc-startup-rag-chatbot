import os
import shutil

# Create json folder if it doesn't exist
os.makedirs("json", exist_ok=True)

# Move all .json files from transcript folder to json folder
for file in os.listdir("transcript"):
    if file.endswith(".json"):
        source_file = f"transcript/{file}"
        dest_file = f"json/{file}"
        shutil.move(source_file, dest_file)
        print(f"Moved {file} to json folder")