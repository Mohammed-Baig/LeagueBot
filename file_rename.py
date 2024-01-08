import os

def rename_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    for file_name in files:
        if file_name.startswith("Screenshot (") and file_name.endswith(").txt"):
            # Extract the number between the parentheses
            number = file_name.split("(")[1].split(")")[0]

            # Generate the new file name
            new_name = f"img_{number}.txt"

            # Build the full paths
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} -> {new_name}")

# Specify the folder path where the files are located
folder_path = "C:/Users/moham/Downloads/text_main"

# Call the function to rename files in the specified folder
rename_files(folder_path)