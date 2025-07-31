from PIL import Image
import os
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoProcessor, AutoModelForCausalLM
from absl import logging
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from utils.img_utils import masks_to_bboxes
from robokit.utils import annotate, overlay_masks
from collections import Counter

def make_gif_from_folder(image_folder, output_gif_path):
    """
    Create an animated GIF from a folder of images.
    
    Args:
        image_folder (str): Path to directory with images.
        output_gif_path (str): Path to save output GIF.
    """
    import glob
    from PIL import Image

    image_files = sorted(glob.glob(f"{image_folder}/*.png"))
    frames = []

    for img_path in image_files:
        img = Image.open(img_path).convert("RGBA")
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        frames.append(bg)

    if frames:
        frames[0].save(
            output_gif_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=150,
            loop=0
        )
        print(f"✅ Saved GIF to: {output_gif_path}")
    else:
        print(f"⚠️ No images found in {image_folder}, GIF not created.")
video_id = "../video50"
rgb_frames_dir = video_id  # directory containing the RGB frames
rgb_gif_path = f"{video_id}.gif"

print(f"🎞️ Creating GIF of RGB frames for {video_id}...")
make_gif_from_folder(rgb_frames_dir, rgb_gif_path)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
video_id = "../video50"
frames_dir = video_id
rgb_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

# === Output folders ===
depth_root = f"{video_id}_depth"
os.makedirs(depth_root, exist_ok=True)

# === Load Depth Anything with use_fast=True ===
print("🌊 Loading Depth Anything V2 Large...")
depth_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Large-hf",
    use_fast=True
)
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
depth_model.to(device).eval()

# === Process all frames ===
print(f"📸 Processing frames in {video_id}...")
for frame_name in tqdm(rgb_frames, desc="🧠 Depth"):
    frame_id = os.path.splitext(frame_name)[0]
    frame_path = os.path.join(frames_dir, frame_name)

    # Load image
    scene_bgr = cv2.imread(frame_path)
    scene_rgb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
    scene_pil = Image.fromarray(scene_rgb)

    # --- Generate depth ---
    inputs = depth_processor(images=scene_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)

    post_processed = depth_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(scene_pil.height, scene_pil.width)],
    )
    depth_tensor = post_processed[0]["predicted_depth"]
    depth_np = depth_tensor.detach().cpu().numpy()

    # Normalize depth to 0–255 for uint8 image
    depth_vis = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    depth_vis_uint8 = (depth_vis * 255).astype(np.uint8)

    # Save normalized depth image
    depth_out_path = os.path.join(depth_root, f"{frame_id}.png")
    cv2.imwrite(depth_out_path, depth_vis_uint8)

print("✅ All normalized depth maps saved.")

depth_dir = f"{video_id}_depth"
depth_gif_path = f"{video_id}_depth.gif"

print(f"🎞️ Creating depth visualization GIF for {video_id}...")
make_gif_from_folder(depth_dir, depth_gif_path)

logging.info("Initialize object detectors")
gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.5) # we set threshold for GroundingDINO here!!!
SAM = SegmentAnythingPredictor(vit_model="vit_h")

def get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, text_prompt='objects', visualize=False):
    """
    Get bounding boxes and masks from gdino and sam
    @param image_path: the image path
    @param gdino: the model of grounding dino
    @param SAM: segment anything model or its variants
    @param text_prompt: generally 'objects' for object detection of noval objects
    @param visualize: if True, visualize the result
    @return: the bounding boxes and masks of the objects.
    Bounding boxes are in the format of [x_min, y_min, x_max, y_max] and shape of (N, 4).
    Masks are in the format of (N, H, W) and the value is True for object and False for background.
    They are both in the format of torch.tensor.
    """
    # logging.info("Open the image and convert to RGB format")
    image_pil = Image.open(image_path).convert("RGB")

    logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
    with torch.no_grad():
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

        # logging.info("GDINO post processing")
        w, h = image_pil.size  # Get image width and height
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
    masks = masks.squeeze(1)
    accurate_bboxs = masks_to_bboxes(masks)  # get the accurate bounding boxes from the masks
    accurate_bboxs = torch.tensor(accurate_bboxs)
    bbox_annotated_pil = None
    if visualize:
        logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), accurate_bboxs, gdino_conf, phrases)
        #bbox_annotated_pil.show()
        display(bbox_annotated_pil)
    return accurate_bboxs, masks, bbox_annotated_pil

# === Device setup
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# === Load PromptGen (Florence-2-base-PromptGen v1.5)
model_id = "createveai/Florence-2-base-PromptGen-v1.5"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager").to(device)

render_root = "../renders/"
object_ids = {}

def clean_caption(text):
    text = text.lower().strip()

    # Remove background or camera phrasing
    text = re.sub(r'the\s+\b.*?\b\s+is completely black.*?$', '', text)
    text = re.sub(r'(positioned|set) .*?(frame|camera|image)', '', text)
    text = re.sub(r'in the center.*?$', '', text)

    # Remove descriptive/style tokens
    text = re.sub(r'\b(stylized|simple|small|dark|black background|set against|depicted|rendering|scene|object|icon|silhouette|resembling)\b', '', text)

    # Remove trailing incomplete phrases
    text = re.sub(r'\b(with|against|in|on|at|of|as|by)\b.*$', '', text)

    # Remove stop words and special chars
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'^(a|an|the)\s+', '', text)

    # Final phrase clipping
    tokens = text.split()
    if len(tokens) >= 2:
        return ' '.join(tokens[:5])
    elif len(tokens) == 1:
        return tokens[0]
    else:
        return ""

def is_valid_caption(text):
    return bool(text and len(text.split()) >= 2)

# Process each 'obj_' folder
render_dirs = sorted(
    d for d in os.listdir(render_root)
    if d.startswith("obj_") and os.path.isdir(os.path.join(render_root, d))
)

print("🧠 Generating object_id labels using PromptGen...")
for obj_dir in tqdm(render_dirs):
    folder = os.path.join(render_root, obj_dir)
    images = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith(".png"))
    if not images:
        continue

    captions = []
    for fname in images:
        try:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
        except:
            continue

        # Use CAPTION prompt instruction
        prompt = "<CAPTION>"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, num_beams=3, use_cache=False)

        raw = processor.batch_decode(out, skip_special_tokens=False)[0]
        # postprocess for prompt-cue
        parsed = processor.post_process_generation(raw, task=prompt, image_size=(img.width, img.height))[prompt]
        cleaned = clean_caption(parsed)

        if is_valid_caption(cleaned):
            captions.append(cleaned)

    if captions:
        # pick most common caption
        best = Counter(captions).most_common(1)[0][0]
        object_ids[obj_dir] = best

print("\n✅ Final object_ids:")
for k, v in object_ids.items():
    print(f"{k}: {v}")

# === Device and paths
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
video_id = "../video50"
frames_dir = video_id
rgb_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
masks_root = f"{video_id}_mask"
os.makedirs(masks_root, exist_ok=True)

# === Utility: clean object_id for filenames
def safe_filename(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)  # remove punctuation
    text = re.sub(r'\s+', '_', text)      # spaces → underscores
    return text

# === Process all frames
print(f"📸 Processing frames in {video_id}...")
for frame_name in tqdm(rgb_frames, desc="🧠 Generating masks"):
    frame_id = os.path.splitext(frame_name)[0]
    frame_path = os.path.join(frames_dir, frame_name)

    for word in object_ids.values():
        try:
            # Get SAM mask
            accurate_bboxs, masks, vis_img = get_bbox_masks_from_gdino_sam(frame_path, gdino, SAM, text_prompt=word, visualize=False)
            if not masks:
                continue

            # Save first mask
            template_mask = masks[0].cpu().numpy().astype(np.uint8) * 255
            clean_name = safe_filename(word)
            mask_path = os.path.join(masks_root, f"{frame_id}_{clean_name}.png")
            cv2.imwrite(mask_path, template_mask)

        except Exception as e:
            print(f"❌ Error for frame {frame_id}, object '{word}': {e}")
            continue

print("✅ All masks saved.")

video_id = "../video50"
mask_root = f"{video_id}_mask"
output_gif_root = os.path.dirname(video_id)  # Save gifs next to video50

# Collect all mask files
mask_files = [f for f in os.listdir(mask_root) if f.endswith(".png")]

# Extract all object names from mask filenames (frameID_object.png)
object_names = sorted(set(
    f.split("_", 1)[1].replace(".png", "") for f in mask_files
))

print(f"🧠 Found object categories: {object_names}")

for object_name in object_names:
    mask_frames = []
    overlay_frames = []

    # Get all mask files for this object
    object_mask_files = sorted([
        f for f in mask_files if f.endswith(f"_{object_name}.png")
    ])

    for mask_file in object_mask_files:
        frame_id = mask_file.split("_")[0]
        mask_path = os.path.join(mask_root, mask_file)
        rgb_path = os.path.join(video_id, f"{frame_id}.png")

        if not os.path.exists(mask_path):
            continue

        # --- Grayscale mask-only GIF ---
        mask_img = Image.open(mask_path).convert("L")
        mask_frames.append(mask_img)

        # --- Overlay GIF ---
        if not os.path.exists(rgb_path):
            print(f"⚠️ Missing RGB frame: {rgb_path}")
            continue

        rgb_img = Image.open(rgb_path).convert("RGBA")
        mask_np = np.array(mask_img)

        red_overlay_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        red_overlay_np[mask_np > 0] = [255, 0, 0, 100]  # Red with alpha

        red_overlay = Image.fromarray(red_overlay_np, mode="RGBA")
        composite = Image.alpha_composite(rgb_img, red_overlay).convert("RGB")
        overlay_frames.append(composite)

    # Save mask-only GIF
    if mask_frames:
        safe_name = object_name.replace(" ", "_").replace("/", "_")
        gif_path = os.path.join(output_gif_root, f"{safe_name}.gif")
        mask_frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=mask_frames[1:],
            duration=150,
            loop=0
        )
        print(f"✅ Saved mask-only GIF: {gif_path}")
    else:
        print(f"⚠️ No mask frames found for {object_name}.")

    # Save overlay GIF
    if overlay_frames:
        safe_name = object_name.replace(" ", "_").replace("/", "_")
        gif_path = os.path.join(output_gif_root, f"{safe_name}_overlay.gif")
        overlay_frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=overlay_frames[1:],
            duration=150,
            loop=0
        )
        print(f"✅ Saved overlay GIF: {gif_path}")
    else:
        print(f"⚠️ No overlay frames found for {object_name}.")