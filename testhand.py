import numpy as np
import os
import cv2

# points = np.load("MP_Data/A/1/2.npy") 
# points = points.reshape(21, 3)

# img = np.zeros((480, 640, 3), dtype=np.uint8)

# for x, y, z in points:
#     cx = int(x * 640)
#     cy = int(y * 480)
#     cv2.circle(img, (cx, cy), 5, (0,255,0), -1)

# cv2.imshow("Keypoints", img)
# cv2.waitKey(0)





# bad_files = 0
# for root, dirs, files in os.walk("MP_Data"):
#     for file in files:
#         data = np.load(os.path.join(root, file))
#         if np.sum(data) == 0:
#             bad_files += 1

# print("Zero frames:", bad_files)

a = np.load("MP_Data/A/1/2.npy")
b = np.load("MP_Data/A/1/3.npy")

print("L2 distance:", np.linalg.norm(a - b))
c = np.load("MP_Data/T/1/2.npy")
print("L2 distance (diff class):", np.linalg.norm(a - c))

