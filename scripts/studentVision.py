import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz, pick_waypoints
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from geometry_msgs.msg import Point

# [x, y] positive x points forward, positive y points leftss
TOP_LEFT = [276.9, 111.8]
TOP_RIGHT = [316.2, -129.5]
BOTTOM_LEFT = [73.7, 30.5]
BOTTOM_RIGHT = [68.6, -22.9]
INIT_POINTS = [TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT, TOP_RIGHT]

# vehicle' initial position [-0.0524168, 0.0147471658]
# topleft       [109, 44]
# bottomleft    [29, 12]
# bottomright   [27, -9]
# topright      [124.5, -51]


class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # 
        
        self.sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, self.img_callback, queue_size=1)

        ## 0848_clip
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)

        # /zed2/zed_node/rgb/image_rect_color
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.pub_waypoint = rospy.Publisher("/target_waypoint", Point, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.B2L_M = self.create_map()
        self.w = 640
        self.h = 480

    # generate the transfomation matrix from the birds eye image to real world distances
    def create_map(self):
        src = np.array([[self.h, self.w/2],[0, self.w/2],[0,-self.w/2], [self.h, -self.w/2]], np.float32)
        dst = np.array(INIT_POINTS, np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return M
        # pass
    
    # warping any point on birds eye image to real world coordinates relative to the vehicle
    def warpPoint(self, waypoint):
        waypoint[0] = self.h - waypoint[0]
        waypoint[1] += self.w / 2
        src_point = np.array(waypoint, np.float32).reshape(1,1,2)
        local_target_point = cv2.perspectiveTransform(src_point, self.B2L_M)[0][0]
        local_target_x = local_target_point[0]
        local_target_y = local_target_point[1]
        return local_target_x, local_target_y

    def img_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        raw_img = cv_image.copy()
        # cv2.imwrite("outimage.jpg", raw_img)
        # raw_img = self.combinedBinaryImage(raw_img)
        # raw_img = self.color_thresh(raw_img)
        # raw_img,_,_ = self.perspective_transform(raw_img)
        # #raw_img = self.gradient_thresh(raw_img)
        # cv2.imshow("image", raw_img)
        # cv2.imwrite("test.jpg", raw_img)
        # cv2.waitKey(0)
        mask_image, bird_image, waypoint = self.detection(raw_img)
        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
            # Publish image message in ROS
            waypoint_x , waypoint_y = self.warpPoint(waypoint)
            print("waypoints:", waypoint_x, waypoint_y)
            out_waypoint_msg = Point()
            out_waypoint_msg.x = waypoint_x
            out_waypoint_msg.y = waypoint_y
            out_waypoint_msg.z = 0
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)
            self.pub_waypoint.publish(out_waypoint_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to perspective_transformfind derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image
        ksize = (5,5)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(grey_img, ksize, 0, 0)
        dev_x = cv2.Sobel(blurred_img, cv2.CV_8U, 1, 0, ksize=3)
        dev_y = cv2.Sobel(blurred_img, cv2.CV_8U, 0, 1, ksize=3)
        combined_img = cv2.addWeighted(dev_x, 0.5, dev_y, 0.5, 0)
        final_img = np.uint8(np.abs(combined_img))
        _, binary_output = cv2.threshold(final_img, thresh_min, thresh_max, cv2.THRESH_BINARY)
        binary_output = np.uint8(binary_output / binary_output.max())
        ####
        return binary_output


    def color_thresh(self, img, thresh=(200, 255)):
        """
        Convert RGB to HSL and threshold to binary image us ing S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        s_lower = thresh[0]
        s_upper = thresh[1]
        
        hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower = np.array([20, 20, 50])
        higher = np.array([40, 255, 255])  # gazebo threshold (0-80, slower-supper, 0-255)
        binary_output = cv2.inRange(hsl_img, lower, higher)
        binary_output = cv2.inRange(hsl_img, lower, higher)
        binary_output = np.uint8(binary_output / binary_output.max())
        ####

        return binary_output


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        h, w = img.shape
        # gazebo settings
        src = np.array([[0,300],[0,h],[w,h], [w,300]], np.float32)
        dst = np.array([[0,0], [0,h], [w,h], [w,0]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)
        warped_img = cv2.warpPerspective((img), M, (w, h))
        # warped_img *= 255
        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.color_thresh(img)
        # binary_img = self.gradient_thresh(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            waypoint = []
            if ret is not None:
                bird_fit_img, waypoint = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)

            else:
                print("Unable to detect lanes")
            # cv2.imshow("image", combine_fit_img)
            # # cv2.imwrite("test.jpg", raw_img)
            # cv2.waitKey(0)
            return combine_fit_img, bird_fit_img, waypoint


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)