# -*- coding:utf-8 _*-
"""
@Author  : Yu Cheng
@Time    : 2019/12/25 9:35
"""
import glob, cv2, os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)

ImgPaths = glob.glob("./OriImage/*.jpg")
H = np.array([[  -0.31606066,   -1.9361168,   631.35385969],
              [   0.01971609,    0.18515113, -173.82800143],
              [   0.00012182,   -0.00378804,    1.0       ]])

# 相机参数
Dist = np.array([-0.285977053720519, 0.074339925936036, -0.008145684407220, 0.0, 0.0], dtype=np.float32)
K = np.array([[308.7962753712953, 0, 325.1614349338094],
              [0, 309.9281771981168, 278.8305865054944],
              [0, 0, 1]], dtype=np.float32)

for ImgPath in ImgPaths:
	print(ImgPath)
	Img = cv2.imread(ImgPath)
	Img = cv2.resize(Img, (640, 480))
	UndistImg = cv2.undistort(Img, K, Dist)
	WarpedImg = cv2.warpPerspective(UndistImg, H, (1000, 1000))
	# plt.imshow(WarpedImg)
	# 	# plt.show()
	SavePath = ImgPath.replace('OriImage', 'WarpedImg')
	os.makedirs(os.path.dirname(SavePath), exist_ok=True)
	cv2.imwrite(SavePath, WarpedImg)
