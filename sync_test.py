import os

def are_folders_synchronized(folder1, folder2, extension1, extension2):
    # Get the list of files in each folder
    files_folder1 = set([file_name for file_name in os.listdir(folder1) if file_name.endswith(extension1)])
    files_folder2 = set([file_name for file_name in os.listdir(folder2) if file_name.endswith(extension2)])

    # Find missing files in each folder
    missing_in_folder1 = files_folder2 - files_folder1
    missing_in_folder2 = files_folder1 - files_folder2

    if missing_in_folder1 or missing_in_folder2:
        print("Folders are not synchronized. Missing files:")
        print("In folder1:", missing_in_folder1)
        print("In folder2:", missing_in_folder2)
        return False

    return True

# Specify the paths of the two folders and file extensions
folder_path_png = "C:/Users/moham/Downloads/text_main"
folder_path_txt = "C:/Users/moham/Downloads/ss_main"
file_extension_txt = ".txt"
file_extension_png = ".jpg"

# Check if folders are synchronized
synchronized = are_folders_synchronized(folder_path_txt, folder_path_png, file_extension_txt, file_extension_png)

if synchronized:
    print("Folders are synchronized.")
