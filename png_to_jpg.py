from PIL import Image
import os

def convert_png_to_jpg_and_replace(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    for file_name in files:
        if file_name.lower().endswith(".png"):
            # Build the full paths
            png_path = os.path.join(folder_path, file_name)

            # Load the PNG image
            image = Image.open(png_path)

            # Generate the new file name with .jpg extension
            jpg_name = os.path.splitext(file_name)[0] + ".jpg"
            jpg_path = os.path.join(folder_path, jpg_name)

            # Convert and save as JPEG
            image.convert("RGB").save(jpg_path, "JPEG")

            # Remove the original PNG file
            os.remove(png_path)

            print(f"Converted and replaced: {file_name} -> {jpg_name}")

# Specify the folder path where the PNG files are located
folder_path = "C:/Users/moham/PycharmProjects/LeagueBot"

# Call the function to convert PNG to JPEG and replace in the specified folder
convert_png_to_jpg_and_replace(folder_path)
