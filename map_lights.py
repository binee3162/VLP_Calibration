#!/usr/bin/env python2
# @brief produce map with light source frequencies in postprocessing for tango based VLP calibration system 
# @author Robin Amsters and Song Gao
# @bug no known bugs

import copy
import cv2
import yaml
import glob
import matplotlib.pyplot as plt
import numpy as np
import vlp_classes as vlp
import time
import os
import pandas as pd

from shutil import copyfile
from calibration_plotting import plot_spectrum, plot_light_map

############################################## FUNCTIONS ###############################################################

if __name__ == "__main__":

    ####################################################### MAIN ###########################################################


    experiment_folders = [
    #'/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/21041905'
    #'/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421041602/210421041602(1)'
    #  '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421041602/210421041602(2)',
       '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(1)second diergepinlvcuole',
      '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(2)',
      '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(3)',
    '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(1)',
    '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(2)',
    '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(3)diyigedengpailiangbian'
     ] 



    for experiment_folder in experiment_folders[0::]: 
        base_path = experiment_folder
        trajectory_path = base_path + "/poses.txt"
        map_path = base_path + "/map.pgm"
        map_info_path = base_path + "/map.yaml"
        print 'Processing dataset: ', experiment_folder

        # VLP objects
        trajectory_bag = vlp.Bag(trajectory_path)  #使用地址初始化包，包含了很多包处理函数
        transformer = vlp.Transformer()  #坐标变换

        # OpenMV camera
        images_bag_path=base_path + "/output.bag"
        image_bag = vlp.Bag(images_bag_path)
        camera = vlp.calibration()



        # Get map data from files
        with open(map_info_path, 'r') as yaml_file:
            map_info = yaml.load(yaml_file)  
        map_img = cv2.imread(map_path)
        trajectory, trajectory_time = trajectory_bag.my_get_joint_data(trajectory_path)
        #get depth information from depth images
        depth=trajectory_bag.my_get_depth(base_path)
        #get translation and orientation data from pose file
        trajectory = {'x':trajectory[0],'y':trajectory[1],'z':trajectory[2],'roll':trajectory[3],'pitch':trajectory[4],'yaw':trajectory[5],'rotation_matrix':trajectory[6],'depth':depth} 


        # Convert pos to map frame 
        trajectory = transformer.trajectory_to_map(map_info, trajectory)  

        # Get centroid coordinates and detected frequencies
        # spectrum, centroids, image_time = camera.my_get_measurements(image_bag) 
        #to save time using exported images
        spectrum, centroids, image_time =camera.get_measurements_from_DCIM(base_path)


        #if possible get android/imu data from bag and use it as corresponding rotation matrix
        imu_matrixs = image_bag.get_rotation_matrix_from_imu(image_time)
        lights_world, centroids, lights_world_noAmend = camera.pixel_to_world_imu(trajectory, trajectory_time, centroids, image_time, spectrum,
                                                                        imu_matrixs)
        # lights_world, centroids, lights_world_noAmend = camera.pixel_to_world(trajectory, trajectory_time, centroids, image_time, spectrum,
        #                                                                 slack=params['slack'])

        if experiment_folder.split("/")[-2]=='210421061741':
            lights_real={'4000':[2.93, 4.30],'1000':[4.13, 3.07],'2000':[2.98, 1.94],'1375':[1.77, 3.17]}  #for the last package
        else:
            lights_real={'1000':[4.33, 3.39], '2000':[3.15, 2.28],'1375': [1.94, 3.57],'4000': [3.12, 4.68]} #last but one



        #paint lights on map
        lights_index, trajectory_index = trajectory_bag.my_align_time_index(image_time[min(image_time,key=image_time.get)],trajectory_time) 
        figname= 'test_light_map'
        plt.figure(num=figname)
        plt.imshow(map_img,
        extent=(0, map_img.shape[1] * map_info['resolution'], 0, map_img.shape[0] * map_info['resolution']),
        cmap='gray_r', label='map')  # Use reverse colormap so 0 is free space and 100 is shown as black
        plt.scatter(trajectory['x'][trajectory_index[0]:], trajectory['y'][trajectory_index[0]:], s=50, c='0.5', label='robot position')
        

        for i in range(len(lights_world.keys())):
            id = lights_world.keys()[i]
            plt.scatter(lights_world[id]['x'], lights_world[id]['y'], s=50, c='r', label='robot position')
        # for i in range(len(lights_world_noAmend.keys())):
        #     id = lights_world_noAmend.keys()[i]
        #     plt.scatter(lights_world_noAmend[id]['x'], lights_world_noAmend[id]['y'], s=50, c='b', label='robot position')
        for i in lights_real:
            plt.scatter(i[0], i[1], s=50, c='b', label='robot position')
        plt.savefig(base_path + '/light_map.png', dpi=300, bbox_inches='tight')

        lights_index, trajectory_index = trajectory_bag.my_align_time_index(image_time[min(image_time,key=image_time.get)],trajectory_time) 
        plt.figure()
        plot_light_map(experiment_folder,map_img,map_info,trajectory,trajectory_index,lights_world,lights_real,multiple=True)
        plt.show()
