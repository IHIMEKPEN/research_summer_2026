import os
from PIL import Image  # pyright: ignore[reportMissingImports]

input_path = "node structure 1.png"
output_path = "Graphical_Abstract.jpg"

# Open and convert to RGB
img = Image.open(input_path)
if img.mode in ("RGBA", "P"):
    img = img.convert("RGB")
    
# Resize to exact dimensions
img_resized = img.resize((660, 295), Image.Resampling.LANCZOS)

# Compress until size is < 45 KB
for q in range(95, 4, -2):
    img_resized.save(output_path, "JPEG", quality=q)
    size_kb = os.path.getsize(output_path) / 1024
    if size_kb <= 44.5:
        print(f"Success! Saved at {size_kb:.2f} KB with quality {q}")
        break