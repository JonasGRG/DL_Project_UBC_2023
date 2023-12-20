import os
import glob
import re

data_path="data/segmentation_data/"

 # Define a function to extract the numeric ID from the filename
def extract_id(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else None

def get_train_val_files():

    # Path to the folders
    thumbnails_path = data_path + 'train_thumbnails'
    labels_path = data_path + 'labels_np_resized'

    # List and sort the files in each folder
    thumbnails = glob(os.path.join(thumbnails_path, "*.png"))
    labels = glob(os.path.join(labels_path, "*.npy"))

    # Sort the lists using the image_ids sorting key
    thumbnails_sorted = sorted(thumbnails, key=extract_id)
    labels_sorted = sorted(labels, key=extract_id)

    train_files = [{"img": thumb, "label": label} for thumb, label in zip(thumbnails_sorted[:120], labels_sorted[:120])]
    val_files = [{"img": thumb, "label": label} for thumb, label in zip(thumbnails_sorted[121:], labels_sorted[121:])]

    return train_files, val_files
    
