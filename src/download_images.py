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
parser.add_argument('--output_dir', type=str, default='data/poster-detect',
                    help='Base directory to save images into train/val/test splits.')
parser.add_argument('--train_frac', type=float, default=0.8,
                    help='Fraction of samples for training.')
parser.add_argument('--val_frac', type=float, default=0.1,
                    help='Fraction of samples for validation.')
args = parser.parse_args()
n_samples = args.n_samples
output_dir = args.output_dir
train_frac = args.train_frac
val_frac = args.val_frac

df = pd.read_csv('data/file-names.csv')
classes = pd.read_csv("data/classes.csv", header=None, names=["LabelName", "DisplayName"])
poster_label = classes[classes["DisplayName"] == "Poster"]["LabelName"].values[0]
print("Poster label ID:", poster_label)

poster_df = df[(df["LabelName"] == poster_label) & (df["Confidence"] == 1)]
poster_ids = poster_df["ImageID"].unique()
print(f"Found {len(poster_ids)} poster images.")

all_ids = set(df["ImageID"].unique())
non_poster_ids = list(all_ids - set(poster_ids))
print(f"Found {len(non_poster_ids)} non-poster images.")

def download_image(image_id, label, split, base_dir):
    subdir = "poster" if label == 1 else "nonposter"
    download_dir = os.path.join(base_dir, split, subdir)
    os.makedirs(download_dir, exist_ok=True)
    bucket = 'open-images-dataset'
    key = f"train/{image_id}.jpg"

    s3 = boto3.client('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    local_file = os.path.join(download_dir, f"{image_id}.jpg")
    try:
        s3.download_file(bucket, key, local_file)
    except botocore.exceptions.ClientError as e:
        print(f"Failed to download {image_id}.jpg: {e}")

# Sample images and split into train, val, test
poster_sample = random.sample(list(poster_ids), min(n_samples, len(poster_ids)))
nonposter_sample = random.sample(non_poster_ids, min(n_samples, len(non_poster_ids)))

# Shuffle for reproducibility
random.seed(621)
random.shuffle(poster_sample)
random.shuffle(nonposter_sample)

# Compute split indices
p_total = len(poster_sample)
n_total = len(nonposter_sample)
p_train_end = int(p_total * train_frac)
p_val_end = p_train_end + int(p_total * val_frac)
n_train_end = int(n_total * train_frac)
n_val_end = n_train_end + int(n_total * val_frac)

poster_train = poster_sample[:p_train_end]
poster_val = poster_sample[p_train_end:p_val_end]
poster_test = poster_sample[p_val_end:]

nonposter_train = nonposter_sample[:n_train_end]
nonposter_val = nonposter_sample[n_train_end:n_val_end]
nonposter_test = nonposter_sample[n_val_end:]

# Download images into split/class directories
for split_name, samples, label in [
    ("train", poster_train, 1),
    ("val", poster_val, 1),
    ("test", poster_test, 1),
    ("train", nonposter_train, 0),
    ("val", nonposter_val, 0),
    ("test", nonposter_test, 0),
]:
    for image_id in tqdm(samples, desc=f"Downloading {split_name} {'poster' if label==1 else 'nonposter'} images"):
        download_image(image_id, label, split_name, output_dir)