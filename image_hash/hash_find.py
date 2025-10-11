import os
import imagehash
from PIL import Image


def calculate_fuzzy_hash(image_path):
    """
    Calculate the perceptual hash value of an image using imagehash.

    Args:
        image_path (str): Path to the image file.

    Returns:
        imagehash.ImageHash: The perceptual hash of the image.
    """
    with Image.open(image_path) as img:
        # You can choose phash/ahash/dhash/whash; here we use phash
        return imagehash.phash(img)


def find_most_similar_image(new_image_path, image_dir, command_file):
    """
    Find the image most similar to the new image and return the corresponding command (based on imagehash hash distance).

    Args:
        new_image_path (str): Path to the new image to compare.
        image_dir (str): Directory containing reference images.
        command_file (str): Path to the command mapping file.

    Returns:
        str or None: The command associated with the most similar image, or None if no match is found.
    """
    # Calculate the hash value of the new image
    new_image_hash = calculate_fuzzy_hash(new_image_path)

    # Read the command file
    commands = {}
    with open(command_file, "r") as f:
        for line in f:
            image_name, command = line.strip().split(" ", 1)
            commands[image_name] = command

    # Store information about the most similar image
    best_match_image = None
    best_match_distance = None

    # Iterate over all images in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            try:
                image_hash = calculate_fuzzy_hash(image_path)
                # Calculate hash distance; smaller means more similar
                distance = new_image_hash - image_hash
                if (best_match_distance is None) or (distance < best_match_distance):
                    best_match_distance = distance
                    best_match_image = image_name
            except Exception:
                continue

    # Return the command for the most similar image
    if best_match_image:
        return commands.get(best_match_image, None)
    else:
        return None


def get_latest_image(image_dir, prefix="temp_", suffix=".png"):
    """
    Find the most recently created image file in the specified folder.
    The filename format should be like temp_YYYYMMDD_HHMMSS.png.

    Args:
        image_dir (str): Directory to search for images.
        prefix (str): Filename prefix to match.
        suffix (str): Filename suffix to match.

    Returns:
        str or None: Full path to the latest image file, or None if not found.
    """
    latest_image = None
    latest_time = -1
    for fname in os.listdir(image_dir):
        if fname.startswith(prefix) and fname.endswith(suffix):
            # Extract timestamp part
            try:
                # Example: temp_20250729_091411.png
                ts = fname[len(prefix) : -len(suffix)]
                # Convert to timestamp (assume format YYYYMMDD_HHMMSS)
                import datetime

                dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
                timestamp = dt.timestamp()
                if timestamp > latest_time:
                    latest_time = timestamp
                    latest_image = fname
            except Exception:
                continue
    if latest_image:
        return os.path.join(image_dir, latest_image)
    else:
        return None


def start_hash_find():
    """
    Start the hash-based image search process.

    Returns:
        str or None: The command associated with the most similar image, or None if not found.
    """
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build relative paths
    new_image_dir = os.path.join(current_dir, "..", "user", "ocr_output")
    # Set the directory for images to match
    image_dir = "image_hash/image"  # Change to your actual path
    command_file = "image_hash/commands.txt"  # Change to your actual path

    # Find the latest image
    new_image_path = get_latest_image(new_image_dir)
    if not new_image_path:
        print("No latest image found.")
        return

    command = find_most_similar_image(new_image_path, image_dir, command_file)
    return command
