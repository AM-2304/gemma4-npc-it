import cv2
import numpy as np

img_path = '/Users/vasu/.gemini/antigravity/brain/85b4e68b-70f3-4407-9e3b-bbdb7c2727d9/crystal_dome_1781003755617.png'
out_path = '/Users/vasu/Downloads/NPC Live Interactions Gemma4-12B/Gemma4NPC/web_ui/assets/crystal_case.png'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# For glowing objects on black background, luminance works as an excellent alpha mask
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# We can boost the alpha slightly so the glass is more solid
alpha = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

img[:, :, 3] = alpha

cv2.imwrite(out_path, img)
print("Saved transparent crystal dome to", out_path)
