import os
import requests
from tqdm import tqdm
from env import GITHUB_TOKEN
# === CONFIG ===
user = "pitcher69"
repo = "IITISOC"
branch = "main"
base_dataset = "DATA/mustard/video"
subfolders = ["rgb", "mask", "depth"]

# Optionally provide a GitHub token for higher rate limits:

headers = {}

def list_png_files(user, repo, branch, path):
    api_url = (
        f"https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={branch}"
    )
    resp = requests.get(api_url, headers=headers)
    resp.raise_for_status()
    items = resp.json()
    pngs = [item["name"] for item in items
            if item["type"] == "file" and item["name"].lower().endswith(".png")]
    return pngs

def download_folder(path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    png_files = list_png_files(user, repo, branch, path)
    print(f"Found {len(png_files)} PNGs in {path}")
    for fname in tqdm(png_files, desc=f"Downloading {path}"):
        raw_url = (
            f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}/{fname}"
        )
        r = requests.get(raw_url)
        if r.status_code == 200:
            with open(os.path.join(local_dir, fname), "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed: {fname} -> status {r.status_code}")

def download():
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    for sub in subfolders:
        gh_path = f"{base_dataset}/{sub}"
        local_folder = os.path.join("data", sub)
        download_folder(gh_path, local_folder)
