import os
import time
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import glob

client = OpenAI()

def generate_image(prompt: str, size: str = "256x256") -> str | None:
    try:
        resp = client.images.generate(
            prompt=prompt,
            n=1,
            size=size,
        )
        # The URL is in resp.data[0].url
        return resp.data[0].url
    except Exception as e:
        return None

def download_image(url: str, save_path: str) -> bool:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        img.save(save_path)
        return True
    except Exception as e:
        return False

def main():
    OUTPUT_DIR = "/Users/tsigall/repositories/WatchdogAI/data/raw-data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    NUM_ITERATIONS = 20

    # Find the highest existing poster number
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "poster_*.png"))
    print(f"Found {len(existing_files)} existing files")
    start_idx = 1
    if existing_files:
        # Extract numbers from filenames and find the max
        numbers = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
        start_idx = max(numbers) + 1
    print(f"Starting from {start_idx}")

    prompts = [
        "A minimalist movie poster for a sci-fi film, dark background with neon accents",
        "A vintage travel poster for Mars, retro style",
        "A concert poster for a jazz festival, art deco style",
        "A book cover for a mystery novel, dark and moody",
        "A motivational poster with a mountain landscape, inspirational quote",
        "A food festival poster with vibrant colors and typography",
        "A music album cover for an indie rock band",
        "A theater play poster, dramatic lighting and composition",
        "A fashion brand poster, high-end luxury aesthetic",
        "A sports event poster, dynamic and energetic design"
    ]

    current_idx = start_idx
    with tqdm(total=len(prompts) * NUM_ITERATIONS, desc="Generating posters") as pbar:
        for prompt in prompts:
            for _ in range(NUM_ITERATIONS):
                url = generate_image(prompt)
                if not url:
                    continue

                # Use the next available number
                save_path = os.path.join(OUTPUT_DIR, f"poster_{current_idx}.png")
                download_image(url, save_path)
                current_idx += 1
                pbar.update(1)
                time.sleep(1)  # Respect API rate limits

if __name__ == "__main__":
    main()
