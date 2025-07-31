from PIL import Image
import glob

image_folder = "./output/renders"
image_files = sorted(glob.glob(f"{image_folder}/*.png"))
def views():
    # Load and convert each image to RGB (removes transparency)
    frames = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGBA")
        # Fill transparent background with black
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        frames.append(bg)
    
    # Save as animated GIF
    if frames:
        output_path = "./output/rendered_views.gif"
        frames[0].save(output_path, format="GIF", save_all=True,
                       append_images=frames[1:], duration=150, loop=0)
        print(f"GIF saved at: {output_path}")
    else:
        print("No images found to create GIF.")
