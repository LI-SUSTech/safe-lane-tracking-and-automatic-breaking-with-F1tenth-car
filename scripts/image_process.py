import cv2
import numpy as np

class ImageProcess():
    def __init__(self):
        input_image_path = 'test_image/image_9.jpg'
        self.image = cv2.imread(input_image_path)
        self.goal=[[0,0],[0,0]]
    def correct_image(self, img, k_matrix):

        inv_K_matrix = np.linalg.inv(k_matrix)

        corrected_image = cv2.warpPerspective(img, inv_K_matrix, (img.shape[1], img.shape[0]))
        # print('image corrected')
        return corrected_image

    def filter_yellow_color(self, image):
        # 指定模糊核的大小，例如 (5, 5)
        kernel_size = (5, 5)

        # 应用平均滤波器进行模糊处理
        image = cv2.blur(image, kernel_size)
        # 转换图像颜色空间为HSV


        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义黄色的HSV范围
        lower_yellow = np.array([20, 70, 100])
        upper_yellow = np.array([30, 255, 255])

        # 创建一个掩码，只保留在黄色范围内的像素
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 使用掩码过滤图像
        result = cv2.bitwise_and(image, image, mask=mask)

        # 保存处理后的图像
        cv2.imwrite("filter_yellow_color.jpg", result)
        
        return result

    def cut_image(self, image):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # print(height, width)

        h = int(height/3)

        # Define the region of interest (ROI) as the bottom half of the image
        roi = image[350:, :]
        # print("height width:", "280", width)

        cv2.imwrite('bottom_half.jpg', roi)

        return roi

    def bird_eye(self, image):
    # Load the image from the camera feed (you'll need to capture the frame from your camera)
    # For the purpose of this example, let's assume you have the frame in the variable 'frame'
        # You can use cv2.VideoCapture to capture frames from a camera
        length = 400
        height = 200
        src_points = np.array([[160, 0], [image.shape[1] - 160, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], dtype='float32')
        print("src_points", src_points)
        dst_points = np.array([[0, 0], [length, 0], [length, height], [0, height]], dtype='float32')

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation
        bird_eye_view = cv2.warpPerspective(image, M, (length, height))
        cv2.imwrite('bird_eye_view.jpg', bird_eye_view)

        return bird_eye_view

    def find_top_n_non_black_pixels(self, image, n):
        # 寻找非黑色像素点的坐标
        non_black_pixels = np.argwhere(image != 0)
        # 按Y坐标排序（从上到下）
        sorted_pixels = non_black_pixels[np.argsort(non_black_pixels[:, 0])]
        # 获取最靠近顶部的n个像素坐标
        top_n_pixels = sorted_pixels[:n]

        for pixel in top_n_pixels:
            cv2.circle(image, (pixel[1], pixel[0]), 5, (0, 255, 0), -1)  # 用绿色圆圈标记像素
        # 保存带有标记的图像
        cv2.imwrite("top_n_pixal.jpg", image)

        return top_n_pixels, image

    def average_coordinates(self, coordinates):
        # 计算坐标的平均值
        if len(coordinates) == 0:
            return None
        avg_y = np.mean(coordinates[:, 1])
        avg_x = np.mean(coordinates[:, 0])
        
        return avg_x, avg_y
    
    def output(self, image, k_matrix):
        
        try:
            corrected_img=self.correct_image(image,k_matrix)
            filter_ed = self.filter_yellow_color(corrected_img)
            filter_ed = self.cut_image(filter_ed)
            img = self.bird_eye(filter_ed)
            top_n_pixels,labled_img = self.find_top_n_non_black_pixels(img, 10)
            # print(top_n_pixels)
            avg_x, avg_y = self.average_coordinates(top_n_pixels)
            self.goal.append([avg_x,avg_y])
            # if(np.abs(self.goal[-1][0]-self.goal[-2][0])>20):
            #     avg_x, avg_y = self.goal[-2][0],self.goal[-2][1]
        except TypeError:
            print("waypoint lost")
            avg_x = np.average(self.goal[-5:-1][0])
            avg_y = np.average(self.goal[-5:-1][1])


        K = 22 / 190 * 2.54 * 0.01
        L = np.sqrt((200-avg_y)**2 + (200-avg_x)**2) * K
        alpha = np.arctan2(200-avg_y, 200-avg_x)
        # print("cord:", avg_x, avg_y)
        
    
        # print(L, alpha)
        return L, alpha, labled_img
