import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':
    num_lane_point = 10

    # 读取所有图片路径
    tests_path = glob.glob('./images/*.jpg')
    for test_path in tests_path:
        # 保存结果地址
        save_res_path =  test_path.split('.')[0] + '_tracking.jpg'
        img = cv2.imread(test_path)

        ################## lane detection ##############################################
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # binaryzation

        height = 128
        width = 128
        half_width = 64

        # img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_out = img
        # print(img_out.shape)
        plt.imshow(img_out)
        plt.show()
        for i in range(num_lane_point):         # each detected point on the lane
            detect_height = height - 13 - i*10
            detect_row = img_out[detect_height]
            line = np.where(detect_row == 255)  # extract zero pixel's index
            # print(line)
            if len(line[0]):                    # If this row has white pixels
                left_pos = np.min(line[0])
                right_pos = np.max(line[0])
            else:
                left_pos = 0
                right_pos = width - 1
            
            center_left = (left_pos, detect_height)
            center_right = (right_pos, detect_height)
            center = (int((left_pos + right_pos)/2), detect_height)
            if left_pos != 0:   # draw the lane points on the binary image
                img_out = cv2.circle(img_out, center_left, 1, (0, 0, 255), thickness=1)
            if right_pos != width - 1:
                img_out = cv2.circle(img_out, center_right, 1, (0, 0, 255), thickness=1)
            if (left_pos != 0) and (right_pos != width - 1):
                img_out = cv2.circle(img_out, center, 1, (0, 255, 0), thickness=1)
        
        plt.imshow(img_out)
        plt.show()
        # cv2.imwrite(save_res_path, img_out)
