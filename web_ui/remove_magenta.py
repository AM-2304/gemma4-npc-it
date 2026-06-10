import cv2
import numpy as np

img_path = '/Users/vasu/.gemini/antigravity/brain/85b4e68b-70f3-4407-9e3b-bbdb7c2727d9/tavern_bar_1781003768184.png'
out_path = '/Users/vasu/Downloads/NPC Live Interactions Gemma4-12B/Gemma4NPC/web_ui/assets/shop_counter_sketch.png'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

lower_magenta = np.array([200, 0, 200, 255])
upper_magenta = np.array([255, 100, 255, 255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([130, 100, 100])
upper_hsv = np.array([170, 255, 255])
mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
mask = cv2.GaussianBlur(mask, (3, 3), 0)
mask_inv = cv2.bitwise_not(mask)
img[:, :, 3] = mask_inv

cv2.imwrite(out_path, img)
print("Saved transparent bar counter to", out_path)
