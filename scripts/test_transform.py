import torch
import cv2
import numpy as np



















M = np.array([[1.030156947932396, -0.0004549625555665835, -14.68579359575316],
              [0.08242474422222842, 1.009183004904826, -106.0147221798931],
              [1.039729824089656e-05, -4.105434239630408e-07, 1]])

img_path = "/data/duser/bczhou_test/warp_bank_all_files/allfiles/yinhang_pufa_3_paizhao_3.jpg"
img = cv2.imread(img_path)
print("transform", M)
h, w = img.shape[:2]

img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

cv2.imwrite("./debug.jpeg", img)

