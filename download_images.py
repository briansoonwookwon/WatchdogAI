import argparse
import pandas as pd
import os
import boto3
import botocore
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download images for poster vs non-poster classification.')
parser.add_argument('--n_samples', type=int, default=10000,
                    help='Number of images to sample for download in each class.')
args = parser.parse_args()
n_samples = args.n_samples

# Load the CSV files
df = pd.read_csv('images/file-names.csv')
classes = pd.read_csv("images/classes.csv", header=None, names=["LabelName", "DisplayName"])

# Identify the poster label
poster_label = classes[classes["DisplayName"] == "Poster"]["LabelName"].values[0]
print("Poster label ID:", poster_label)  # should be /m/01n5jq

poster_df = df[(df["LabelName"] == poster_label) & (df["Confidence"] == 1)]
poster_ids = poster_df["ImageID"].unique()
print(f"Found {len(poster_ids)} poster images.")

# Non-poster images
all_ids = set(df["ImageID"].unique())
non_poster_ids = list(all_ids - set(poster_ids))
print(f"Found {len(non_poster_ids)} non-poster images.")

def download_image(image_id, label, base_dir="images"):
    """Downloads an image from the S3 bucket."""
    subdir = "poster" if label == 1 else "nonposter"
    download_dir = os.path.join(base_dir, subdir)
    os.makedirs(download_dir, exist_ok=True)
    bucket = 'open-images-dataset'
    key = f"train/{image_id}.jpg"

    s3 = boto3.client('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    local_file = os.path.join(download_dir, f"{image_id}.jpg")
    try:
        s3.download_file(bucket, key, local_file)
    except botocore.exceptions.ClientError as e:
        print(f"Failed to download {image_id}.jpg: {e}")

# Sample images for downloading
poster_sample = random.sample(list(poster_ids), min(n_samples, len(poster_ids)))
nonposter_sample = random.sample(non_poster_ids, min(n_samples, len(non_poster_ids)))

for image_id in tqdm(poster_sample, desc="Downloading poster images"):
    download_image(image_id, label=1)

for image_id in tqdm(nonposter_sample, desc="Downloading non-poster images"):
    download_image(image_id, label=0)