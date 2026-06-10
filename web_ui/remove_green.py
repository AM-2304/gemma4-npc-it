import cv2
import numpy as np

# Load the image
img_path = '/Users/vasu/.gemini/antigravity/brain/85b4e68b-70f3-4407-9e3b-bbdb7c2727d9/shop_counter_green_1781002376179.png'
out_path = '/Users/vasu/Downloads/NPC Live Interactions Gemma4-12B/Gemma4NPC/web_ui/assets/shop_counter_sketch.png'

# Read with alpha channel
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img.shape[2] == 3:
    # Add alpha channel if missing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# Define the green color range to remove
# The background is bright pure green, e.g. RGB(0, 255, 0)
lower_green = np.array([0, 200, 0, 255])
upper_green = np.array([100, 255, 100, 255])

# Convert to HSV for better masking
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([40, 50, 50])
upper_hsv = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

# Invert mask (we want to keep non-green pixels)
mask_inv = cv2.bitwise_not(mask)

# Set alpha channel to 0 where mask is true
img[:, :, 3] = mask_inv

# Save the resulting image
cv2.imwrite(out_path, img)
print("Saved transparent shop counter to", out_path)
