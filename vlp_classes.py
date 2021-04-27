#coding=utf-8
#!/usr/bin/env python2
# @brief VLP class definitions
# @details Not intended for simulations
# @author Robin Amsters
# @bug No known bugs
import copy
import cv2
import rospy
import yaml
import rosbag
import glob
import numpy as np
import scipy.stats as sc
import time as times
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_matrix
from scipy.signal import butter, lfilter



def get_yaml_params(yaml_file):
    """!
        @brief get parameters from YAML file
        @param yaml_file: String containing the full path to the YAML file
        @return params: parameters from the YAML file returned as a dictionary
    """
    with open(yaml_file, 'r') as yaml_file:
        params = yaml.load(yaml_file)

    # Convert lists to numpy matrices if they are specified in the 'matrices' field
    for key in params.keys():
        param_group = params[key]

        try:
            if type(param_group) == dict:  # Only loop over dictionaries
                for param in param_group.keys():
                    if param in params['matrices']:
                        param_group[param] = np.asmatrix(param_group[param])

        except KeyError: # Only do this when matrices are defined in the yaml file
            pass

    yaml_file.close()

    return params

class Bag():
    """!
     @brief Rosbag processing class
     @details provides additional functionality to the rosbag.Bag class
    """

    def __init__(self, file_path=""):

        ## Complete path to file (as a string)
        self.file_path = file_path

    def get_topic_data(self, topic):
        """!
            @brief Return all messages from a specific topic

            @param topic: ROS topic present in rosbag file

            @return all_msg: list containing all message
            @return all_t: timestamp of messages in all_msg
        """

        all_msg = []
        all_t = []

        # Initialize rosbag object
        bag = rosbag.Bag(self.file_path)
        for topic, msg, t in bag.read_messages(topics=[topic]):
            all_msg = np.append(all_msg, msg)
            t = msg.header.stamp
            all_t = np.append(all_t, t.to_sec())

        return all_msg, all_t
    def get_rotation_matrix_from_imu(self, t ):
        """!
            @brief Acquire rotation_matrix from quaternion of imu data

            @param t: timestamps of the imu data required

            @return matrixs: rotation matrixs in a dict with t as key
        """
        unique_time=np.unique(t.values())
        value=np.zeros(unique_time.shape)
        bag = rosbag.Bag(self.file_path)
        matrixs=dict(zip(unique_time,value))
        for stamp in unique_time:
            
            for topic, msg, t in bag.read_messages(topics='/android/imu'):
                
                t = msg.header.stamp.to_sec()
                if t>stamp:
                    if abs(last_t-stamp)>abs(t-stamp):
                        quaternion=msg.orientation
                        break
                    else:
                        quaternion=last_msg.orientation
                        break
                last_msg=msg
                last_t = t
            rotation_matrix= quaternion_matrix(np.array([quaternion.x, quaternion.y,quaternion.z,quaternion.w]))
            matrixs[stamp]=rotation_matrix
        return matrixs
 
    def my_get_depth(self, file_name):
        """!
            @brief get depth info from depth images

            @param filename: experimental folder

            @return depth: array of all the depth data in this folder
        """
        root_folder=file_name+'/depth'
        image_paths = glob.glob(root_folder + "/*.png")
        depth = np.array([])
        for image_path in image_paths:
            image=cv2.imread(image_path,cv2.IMREAD_ANYDEPTH)
            shape=image.shape
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            centerPoint={shape[0]/2,shape[1]/2}
            center=image[62:72,115:125]
            sum=0
            len=0
            for i in center:
                for j in i:
                    if j!=0:
                        sum+=j
                        len+=1
            if len!=0:
                average=sum/len
            else:
                average=0
            depth=np.append(depth,average*0.001)
        return depth
    def my_get_joint_data(self, file_name):
        """!
            @brief get translation and orientation data from pose.txt. Note that if imu data is avaliable the rotation matrixs extract here will be discarded.
            @param file_name: path of pose file 
            @return pose: list containing the pose of the frame [x, y, z,roll, pitch, yaw,rotation_matrixs]. Coordinates are sublists
            @return all_t: timestamp of tf messages
        """
        x = np.array([])
        y = np.array([])
        z = np.array([])
        roll=np.array([])
        pitch=np.array([])
        yaw=np.array([])
        rotation_matrixs=np.array([])  #with offset
        all_t = np.array([])
        number=[]
        with open(file_name, 'r') as f:
            for line in f:
                number.extend([item
                for item in line.split()
                ])
        index=0
        while index<len(number):
            all_t=np.append(all_t,float(number[index]))
            x=np.append(x,float(number[index+1]))
            y=np.append(y,float(number[index+2]))
            z=np.append(z,float(number[index+3]))
            euler = euler_from_quaternion(
                [float(number[index+4]), float(number[index+5]), float(number[index+6]), float(number[index+7])])
            roll=np.append(roll, euler[0])
            pitch=np.append(pitch,euler[1])
            yaw=np.append(yaw, euler[2])
            rotation_matrix= np.matrix(quaternion_matrix(np.array([float(number[index+4]), float(number[index+5]), float(number[index+6]), float(number[index+7])])))
            rotation_matrixs=np.append(rotation_matrixs,rotation_matrix)
            index+=8        
        pose = [x, y, z, roll, pitch, yaw,rotation_matrixs]
        return pose, all_t


    def my_align_time_index(self, light_time, traj_time): 
        """!
            @brief return indexes of two time series for which their timestamps are approximately equal.

        """
        index_1 = []
        index_2 = []
        for i in range(len(light_time)):
            time1=light_time[i]
            for j in range(len(traj_time)):
                time2=traj_time[j]
                if time2>time1:
                    index_1.append(i)
                    index_2.append(j-1)
                    break
        return index_1,index_2

      

class Transformer():
    """!
    @brief Transformer class, performs coordinate transformations for individual points and arrays.
    """


    def trajectory_to_map(self, map_info, trajectory):
        """!
        @brief convert trajectory from cartographer to map frame

        @param map_msg: nav_msgs/OccupancyGrid message
        @param trajectory: marker points extracted from MarkerArray message

        @return: trajectory, translated to the map origin
        """
        origin = map_info['origin']  #eg.origin: [-3.44038, -1.60912, 0.0] 

        trajectory['x'] = np.subtract(trajectory['x'], origin[0])
        trajectory['y'] = np.subtract(trajectory['y'], origin[1])

        try:
            trajectory['z'] = np.subtract(trajectory['z'], origin[2])
        except KeyError:
            pass

        return trajectory


class camera(object):
    """!
    @brief Class to be used for processing images 
    """

    def __init__(self, d_cam=0.287, dz = 1.5, z_cam=0.23, f=2.8*10**(-3), x_range=200, y_range=200, center=(327, 198), line_pxl = 784.0,
                 clk_freq=9.0, pll=6.0, mode = 2.0, time_scale = 1000000.0, pix_size=6*10**-6, pix_h=480, pix_w=640,
                 camera_matrix = np.array([[1.61427666e+03, 0.00000000e+00, 2.66426581e+02],
                                 [0.00000000e+00, 1.63637306e+03, 1.85463848e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                 dist_coeffs = np.array([-2.54438196e+00, 2.69423578e+01,
                                        -2.59202859e-02, 6.46769808e-02, -1.51625048e+02]),
                 light_map={1565.4: np.array([0.0, 0.0, 1.5]),
                       2025.8: np.array([0.0, 1.618, 1.5]),
                       2869.9: np.array([1.782, 1.629, 1.5]),
                       4919.8: np.array([1.782, 0.0, 1.5])},
                 vertical_kernel=(25,45),
                 horizontal_kernel=(185, 1),
                 threshold= 90,
                 eval_area=(70,40),
                 canny_thresholds = (75,145)
                 ):
        #@todo check which variables are really needed

        ## Distance to robot center [m]
        self.d_cam = d_cam

        ## Heigth difference between camera and floor [m]
        self.z_cam = z_cam

        ## Heigth difference between camera and lights [m]
        self.dz = dz

        ## Focal length [m]
        self.f = f

        ## Range of x-coordinates around the center where we should look for lights [pixels]
        self.x_range = x_range

        ## Range of y-coordinates around the center where we should look for lights (in pixels)
        self.y_range = y_range

        ## camera optical center (x,y)
        self.center = center

        ## Heigth of an image [px]
        self.pix_h = pix_h

        ## Width of an image [px]
        self.pix_w = pix_w

        ## Camera matrix
        self.camera_matrix = camera_matrix

        ## Distortion coefficients
        self.dist_coeffs = dist_coeffs

        ## Location and frequency of lights in the world frame
        self.light_map = light_map

        ## Total line length in pixels
        self.line_pxl = line_pxl

        ## Clock frequency
        self.clk_freq = clk_freq

        ## Internal Clk Frequency upscaling
        self.pll = pll

        ## Division of effective Clk frequency due to YUV422 output
        self.mode = mode

        ## Time unit conversion us -> s
        self.time_scale = time_scale

        ## Pixel size for vertical and horizontal direction [m]
        self.pix_size = pix_size

        ## line readout time
        self.line_time = line_pxl / (clk_freq * pll / mode * time_scale)  # line time, expressed in us: 784/27 = 29.037us

        ## Kernel for blurring to remove stripe pattern
        self.vertical_kernel=vertical_kernel

        ## Horizontal blurring kernel to smear out stripes
        self.horizontal_kernel = horizontal_kernel

        ## Threshold for binary image
        self.threshold = threshold

        ## Evaluation area for frequency detection
        self.eval_area = eval_area

        ## Thresholds for canny edge detection
        self.canny_thresholds = canny_thresholds

        cameraMtx2, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (self.pix_w , self.pix_h), 0, (self.pix_w , self.pix_h))
        mapx, mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, cameraMtx2, (self.pix_w , self.pix_h), 5)

        ## Mapping for image distortion removal [mapx, mapy]
        self.rectify_map = (mapx, mapy)

    def check_image_center(self, light_coordinate):
        """!
        @brief check if a detected centroid is approximately in the optical center
        @details uses the properties x_range and y_range

        @param light_coordinate: pixel coordinates of the detected centroid

        @return True if the detected centroid is in the defined range, False otherwhise
        

        """
        # x_min = self.center[0] - self.x_range
        # x_max = self.center[0] + self.x_range
        # y_min = self.center[1] - self.y_range
        # y_max = self.center[1] + self.y_range
        # x_min = 1632 - self.x_range
        # x_max = 1632 + self.x_range
        # y_min = 1224 - self.y_range
        # y_max = 1224 + self.y_range
        x_min = 960 - 100
        x_max = 960 + 100
        y_min = 540 - 100
        y_max = 540 + 100
        if ((x_min <= light_coordinate[0] <= x_max) and (y_min < light_coordinate[1] < y_max)):
            return True
        else:
            return True


    def draw_contours(self, image_in):
            """!
            @brief Function processes a greyscale image to find bright blobs (lights) and determine their centroids &
            coordinates in the image.
            @details Returns a list of centroid coordinates (one coordinate tuple per centroid) and an enclosing radius.
            @param image_in: image to process, should be preprocessed first
            @return coordinate_l: pixel coordinates of the detected centroid
            @return radius_l: radius of detected centroid
            """

            coordinate_l = []
            radius_l = []
            if cv2.__version__[0] == str(4):
                contours, hierarchy = cv2.findContours(image_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cv2.__version__[0] == str(3):
                im2, contours, hierarchy = cv2.findContours(image_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = np.zeros_like(image_in) 

            cv2.drawContours(out, contours, -1, 255, 3) 
            path='/home/song/Desktop/catkin_ws/contour.jpeg'
            cv2.imwrite(path,out)
            index = 1
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                coordinate = (int(cX), int(cY)) #contours 中心
                coordinate_l.append(coordinate)
                (x, y), radius = cv2.minEnclosingCircle(c)
                radius_l.append(radius)
                index += 1
            return coordinate_l, radius_l

    def img_preprocess(self, image_in):
        """!
        @brief Performs image processing functions on input image to determine centroids of light transmitters (blurring and
        @details Returns thresholded image for localisation and list containing a coordinate tuple and radius for each
        light. As a precaution, it checks whether the the input image is already greyscaled, and performs Greyscale conversion
        if not. 
        @param image_in: image to process
        @return threshold_image: Image that is Gaussian blurred and binary thresholded
        @return coordinates: Centroid coordinates ?
        @return radius: Centroid radius ?
        """
        if len(image_in.shape) > 2:  # "Shape" of colour image has 3 dimensions, only 2 for greyscale
            grey_image = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY) # Greyscale the input image only if input image is colour image
        else:
            grey_image = image_in
        blur_image = cv2.GaussianBlur(grey_image, self.vertical_kernel, 0) # (25, 35) #OV7725     #(125,175) #IMX129
        ret, threshold_image = cv2.threshold(blur_image, self.threshold, 255, cv2.THRESH_BINARY) # 95 'OV7725     #55 #IMX219 
        coordinates, radius = self.draw_contours(threshold_image)
        return threshold_image, coordinates, radius
    def my_get_frequency(self, image_in, coordinate,radius):
        """!
        @brief Processes image for frequency calculation.
        @details Takes image array and centroid coordinate as inputs. Selects sub-image
        around coordinates according to internal parameters eval_x and eval_y.
        Calculating the average pixels without edge detection. Returns frequency [Hz] and averaged stripe height [pixels}.
        @param coordinate: centroid coordinates (x,y) [m]
        @return light_frequency: frequency of detected light source [Hz]
        @return average: 
        """
        eval_x = int(radius*0.5)
        eval_y = int(radius*0.5)
        eval_area = [eval_x, eval_x, eval_y, eval_y]  # evaluation in -x, +x, -y & +y from centroid
        x = coordinate[0]
        y = coordinate[1]

        im2 = image_in[(y - eval_area[2]):(y + eval_area[3]), (x - eval_area[0]):(x + eval_area[1])]
        grey_image = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        ret, threshold_rect_image = cv2.threshold(grey_image, 140, 255, cv2.THRESH_BINARY)
        x = eval_area[0]
        y = eval_area[2]
        x_eval = x - eval_area[0]
        white_pixels = []
        black_pixels = []
        while x_eval < (x + eval_area[1]):
            dmc1 = threshold_rect_image[y - eval_area[2]: y + eval_area[3], x_eval: x_eval + 1]  #一列
            index = 0
            first=dmc1[index]
            while dmc1[index]==first and index<len(dmc1)-1:
                index=index+1
            front=index
            black_count=0
            black_sums=0
            white_count=0
            white_sums=0
            while index < len(dmc1)-1:
                if dmc1[front]!=dmc1[index]:
                    if dmc1[front]==255:
                        black_count+=1
                        black_sums+=index-front
                    else:
                        white_count+=1
                        white_sums+=index-front
                    front=index
                index+=1
            if white_count!=0:
                white_pixels.append(float(white_sums)/float(white_count))
            if black_count!=0:
                black_pixels.append(float(black_sums)/float(black_count))
            x_eval += 1
        if len(white_pixels) != 0 and len(black_pixels) != 0:
            average = sum(white_pixels)/len(white_pixels)+sum(black_pixels)/len(black_pixels)
            light_period = average  * 22.7272727272  #rolling shutter line time of resolution 1920*1080 in us
            light_frequency = 1000000.0 / light_period
        else:
            average = 0.0
            light_frequency = 0.0
        return light_frequency, average

class calibration(camera):
    """!
    Camera class that is used specifically for calibration
    """
    def get_measurements_from_DCIM(self, path):
         """!
        @brief process images in folder to obtain calibration data
        @param path: complete path to the folder that has image messages of stripe pattern LEDs
        @return spectrum: complete spectrum
        @return light_id: identified frequencies with number of occurences
        @return centroids: detected light locations
        """
        DCIM=path+'/DCIM'
        image_paths = glob.glob(DCIM + "/*.jpg")
        centroids = {}
        time = {}
        frequencies = np.zeros(len(image_paths))

        for i in range(len(image_paths)):
            t=image_paths[i][-19:-4]
            timestamp=times.mktime(times.strptime(t,"%Y%m%d_%H%M%S"))
            image= cv2.imread(image_paths[i])
            im2, coordinate_l, radius_l = self.img_preprocess(image)
            for index in range(len(coordinate_l)): # @todo check index use, current function might only work if 1 light source is in the image
                if radius_l[index]!=max(radius_l):
                    continue
                light_frequency, average = self.my_get_frequency(image, coordinate_l[index],radius_l[index])#customed 
                if light_frequency != 0:
                    frequencies[i] =  light_frequency

                    # Add centroid location to frequency based dictionary
                    try:
                        centroids[light_frequency]['x'] = np.append(centroids[light_frequency]['x'], coordinate_l[0][0])
                        centroids[light_frequency]['y'] = np.append(centroids[light_frequency]['y'], coordinate_l[0][1])
                        time[light_frequency] = [timestamp]

                    # Create new entry in dictionary if needed
                    except KeyError:
                        centroids[light_frequency] = {}
                        centroids[light_frequency]['x'] = np.array([coordinate_l[0][0]])
                        centroids[light_frequency]['y'] = np.array([coordinate_l[0][1]])
                        time[light_frequency] = [timestamp]
                break

        # Count occurrences of unique frequencies
        unique_frequencies, frequency_counts = np.unique(frequencies, return_counts=True)
        frequencies_dict = dict(zip(unique_frequencies, frequency_counts))
        return frequencies_dict, centroids, time
    def my_get_measurements(self, bag):
        """!
        @brief process images in a rosbag file to obtain calibration data
        @param bag: complete path to a rosbag file that has image messages of stripe pattern LEDs
        @return spectrum: complete spectrum
        @return light_id: identified frequencies with number of occurences
        @return centroids: detected light locations
        """

        bridge = CvBridge()  # Converter for ROS to OpenCV images

        # Get messages from bag file
        image_msgs, image_time = bag.get_topic_data(topic='/test/frontCamera')

        # Initialize collections
        centroids = {}
        time = {}
        frequencies = np.zeros(len(image_msgs))

        # Detect frequencies and centroid coordinates in each image
        for i in range(len(image_msgs)):
            cv_image = bridge.compressed_imgmsg_to_cv2(image_msgs[i], desired_encoding="bgr8")
            image = cv_image
	        # cv2.imwrite("~/Desktop/catkin_ws",image)
            im2, coordinate_l, radius_l = self.img_preprocess(image)
            # Detect frequency and light source for each detected light source
            for index in range(len(coordinate_l)): # @todo check index use, current function might only work if 1 light source is in the image

                if self.check_image_center(coordinate_l[index]):
                    light_frequency, average = self.my_get_frequency(image, coordinate_l[index],radius_l[index])#customed 
                    if light_frequency != 0:
                        frequencies[i] =  light_frequency

                        # Add centroid location to frequency based dictionary
                        try:
                            centroids[light_frequency]['x'] = np.append(centroids[light_frequency]['x'], coordinate_l[0][0])
                            centroids[light_frequency]['y'] = np.append(centroids[light_frequency]['y'], coordinate_l[0][1])
                            time[light_frequency] = np.append(time[light_frequency], image_time[i])

                        # Create new entry in dictionary if needed
                        except KeyError:
                            centroids[light_frequency] = {}
                            centroids[light_frequency]['x'] = np.array([coordinate_l[0][0]])
                            centroids[light_frequency]['y'] = np.array([coordinate_l[0][1]])
                            time[light_frequency] = np.array([image_time[i]])

        # Count occurrences of unique frequencies
        unique_frequencies, frequency_counts = np.unique(frequencies, return_counts=True)
        frequencies_dict = dict(zip(unique_frequencies, frequency_counts))
        return frequencies_dict, centroids, time
    def pixel_to_world_imu(self, trajectory, trajectory_time, all_lights_pixel, lights_time, spectrum, imu_matrixs):
        """
        @brief convert detected centroids (in pixels) to world coordinates 
        @details using imu based roation matrixs
        @param trajectory: SLAM trajectory
        @param trajectory_time: timestamps corresponding to entries in trajectory
        @param all_lights_pixel: centroid coordinates as returned by get_measurements
        @param lights_time: timestamps of the centroid coordinates
        @param slack: allowed mismatch between timestamps [s]
        @return lights_world: light coordinates in the world frame
        """
        # Copy dictionaries to avoid edition them
        trajectory_orig = copy.deepcopy(trajectory)
        all_lights_pixel = copy.deepcopy(all_lights_pixel)
        lights_time = copy.deepcopy(lights_time)
        imu_matrixs = copy.deepcopy(imu_matrixs)
        # Initialize collections
        lights_world = dict.fromkeys(all_lights_pixel.keys())  #centroid的key应该是对应的文件编号
        all_camera_pose = dict.fromkeys(all_lights_pixel.keys())
        lights_world_noAmend=dict.fromkeys(all_lights_pixel.keys())
        # Convert to world frame for every identified light source
        for id in all_lights_pixel.keys(): #按照不同照片进行处理
            trajectory = copy.deepcopy(
                trajectory_orig)  # Trajectory will be timesynched per light source, so initialize new version
            lights_world[id] = {'x': np.array([]), 'y': np.array([])}
            lights_world_noAmend[id] = {'x': np.array([]), 'y': np.array([])}
            lights_pixel = all_lights_pixel[id]

            # Align centroid coordinates and trajectory timestamps
            bag_sync = Bag()
            lights_index, trajectory_index = bag_sync.my_align_time_index(lights_time[id],trajectory_time) 
            for coordinate in trajectory.keys(): #y x theta
                if coordinate=='rotation_matrix':
                    #trajectory[coordinate] = trajectory[coordinate][(16*trajectory_index[0]):(16*trajectory_index[0]+16)]
                    continue
                trajectory[coordinate] = trajectory[coordinate][trajectory_index]#只保留关键帧
                
                    

            for coordinate in lights_pixel.keys():
                lights_pixel[coordinate] = lights_pixel[coordinate][lights_index]

            # Remove entries that were filtered by time synchronisation
            if len(lights_pixel['x']) == 0 and len(lights_pixel['y']) == 0:
                all_lights_pixel.pop(id)
                lights_world.pop(id)
                all_camera_pose.pop(id)
                spectrum.pop(id)

            else:

                dx = -(lights_pixel['x'][0] - 960)
                dy = (lights_pixel['y'][0] - 540)
                dz = (1.5-trajectory['depth'][0])#根据depthcamera



                x = -dx * (1.4*10**(-6)) * abs(dz) / (3.75*10**(-3))  #其实这里dz有一定误差，因为手机不是完全水平的，暂时不知道有什么合适的办法
                y = -dy * (1.4*10**(-6)) * abs(dz) / (3.75*10**(-3))
                z=dz


                rotation_matrix=imu_matrixs[lights_time[id][0]]
                #z方向没有bias，需要加上离地面的高度，
                rotation_matrix[0][3]=trajectory['x'][0]
                rotation_matrix[1][3]=trajectory['y'][0]
                rotation_matrix[2][3]=trajectory['depth'][0]
                results= np.array(np.dot(rotation_matrix,  np.array([x,y,z,1]).reshape(4,1)))
                lights_world[id]['x']=results[0]
                lights_world[id]['y']=results[1]
                lights_world_noAmend[id]['x']=trajectory['x'][0]
                lights_world_noAmend[id]['y']=trajectory['y'][0]                


        return lights_world, all_lights_pixel,lights_world_noAmend
    def pixel_to_world(self, trajectory, trajectory_time, all_lights_pixel, lights_time, spectrum, slack=0.01):
        """
        @brief convert detected centroids (in pixels) to world coordinates
        @param trajectory: SLAM trajectory
        @param trajectory_time: timestamps corresponding to entries in trajectory
        @param all_lights_pixel: centroid coordinates as returned by get_measurements
        @param lights_time: timestamps of the centroid coordinates
        @param slack: allowed mismatch between timestamps [s]

        @return lights_world: light coordinates in the world frame
        """
        # Copy dictionaries to avoid edition them
        trajectory_orig = copy.deepcopy(trajectory)
        all_lights_pixel = copy.deepcopy(all_lights_pixel)
        lights_time = copy.deepcopy(lights_time)

        # Initialize collections
        lights_world = dict.fromkeys(all_lights_pixel.keys())  #centroid的key应该是对应的文件编号
        all_camera_pose = dict.fromkeys(all_lights_pixel.keys())
        lights_world_noAmend=dict.fromkeys(all_lights_pixel.keys())
        # Convert to world frame for every identified light source
        for id in all_lights_pixel.keys(): #按照不同照片进行处理
            trajectory = copy.deepcopy(
                trajectory_orig)  # Trajectory will be timesynched per light source, so initialize new version
            lights_world[id] = {'x': np.array([]), 'y': np.array([])}
            lights_world_noAmend[id] = {'x': np.array([]), 'y': np.array([])}
            lights_pixel = all_lights_pixel[id]

            # Align centroid coordinates and trajectory timestamps
            bag_sync = Bag()
            #修改这个函数!使其返回左侧的trajectory
            lights_index, trajectory_index = bag_sync.my_align_time_index(lights_time[id],trajectory_time) 
            for coordinate in trajectory.keys(): #y x theta
                if coordinate=='rotation_matrix':
                    trajectory[coordinate] = trajectory[coordinate][(16*trajectory_index[0]):(16*trajectory_index[0]+16)]
                    continue
                trajectory[coordinate] = trajectory[coordinate][trajectory_index]#只保留关键帧
                
                    

            for coordinate in lights_pixel.keys():
                lights_pixel[coordinate] = lights_pixel[coordinate][lights_index]

            # Remove entries that were filtered by time synchronisation
            if len(lights_pixel['x']) == 0 and len(lights_pixel['y']) == 0:
                all_lights_pixel.pop(id)
                lights_world.pop(id)
                all_camera_pose.pop(id)
                spectrum.pop(id)

            else:

                dx = -(lights_pixel['x'][0] - 960)
                dy = (lights_pixel['y'][0] - 540)
                dz = (1.5-trajectory['depth'][0])#根据depthcamera

                x = -dx * (1.4*10**(-6)) * abs(dz) / (3.75*10**(-3))  
                y = -dy * (1.4*10**(-6)) * abs(dz) / (3.75*10**(-3))
                z=dz

                rotation_matrix=np.array(trajectory['rotation_matrix']).reshape(4,4)
                #z方向没有bias，需要加上离地面的高度，
                rotation_matrix[0][3]=trajectory['x'][0]
                rotation_matrix[1][3]=trajectory['y'][0]
                rotation_matrix[2][3]=trajectory['depth'][0]
                results= np.array(np.dot(rotation_matrix,  np.array([x,y,z,1]).reshape(4,1)))
                lights_world[id]['x']=results[0]
                lights_world[id]['y']=results[1]
                lights_world_noAmend[id]['x']=trajectory['x'][0]
                lights_world_noAmend[id]['y']=trajectory['y'][0]                

        return lights_world, all_lights_pixel,lights_world_noAmend

