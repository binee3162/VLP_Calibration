#!/usr/bin/env python2
# @brief plot map with light source frequencies.Note that the lights coordinate is extracted using maplights.py
# @author Song Gao

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
from calibration_plotting import plot_spectrum, plot_light_map, plot_all_light_map


if __name__ == "__main__":

    ####################################################### MAIN ###########################################################



    fig=plt.figure("all")
    subIndex=1
    experiment_folders = [
    #'/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/21041905'
    #'/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421041602/210421041602(1)'
    #  #'/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421041602/210421041602(2)',
       '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(1)second diergepinlvcuole',
      '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(2)',
      '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(3)',
    '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(1)',
    '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(2)',
    '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(3)diyigedengpailiangbian'
     ] 



    for experiment_folder in experiment_folders[0::]: 
        t_start = time.time()
        base_path = experiment_folder
        trajectory_path = base_path + "/poses.txt"
        map_path = base_path + "/map.pgm"
        map_info_path = base_path + "/map.yaml"
        print 'Processing dataset: ', experiment_folder

        # VLP objects
        trajectory_bag = vlp.Bag(trajectory_path)  #使用地址初始化包，包含了很多包处理函数
        transformer = vlp.Transformer()  #坐标变换


        images_bag_path=base_path + "/output.bag"
        image_bag = vlp.Bag(images_bag_path)
        camera = vlp.calibration()



        # Get data from files
        with open(map_info_path, 'r') as yaml_file:
            map_info = yaml.load(yaml_file)  
        map_img = cv2.imread(map_path)
        trajectory, trajectory_time = trajectory_bag.my_get_joint_data(trajectory_path)
        depth=trajectory_bag.my_get_depth(base_path)
        trajectory = {'x':trajectory[0],'y':trajectory[1],'z':trajectory[2],'roll':trajectory[3],'pitch':trajectory[4],'yaw':trajectory[5],'rotation_matrix':trajectory[6],'depth':depth} #key是x y theta


        # Convert robot pos to map frame
        trajectory = transformer.trajectory_to_map(map_info, trajectory) 
        spectrum, centroids, image_time =camera.get_measurements_from_DCIM(base_path)


        
        if experiment_folder.split("/")[-2]=='210421061741':
            lights_real={'4000':[2.93, 4.30],'1000':[4.13, 3.07],'2000':[2.98, 1.94],'1375':[1.77, 3.17]}  #for the last package
        else:
            lights_real={'1000':[4.33, 3.39], '2000':[3.15, 2.28],'1375': [1.94, 3.57],'4000': [3.12, 4.68]} #last but one
        lights_world={
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(1)second diergepinlvcuole':{1011.4942528768: {'x': [4.30477277], 'y': [3.26177099]}, 1404.255319153428: {'x': [1.9275354], 'y': [3.51809512]}, 2019.672131154004: {'x': [3.1969149], 'y': [2.26302395]}, 4082.4742268171817: {'x': [3.18190249], 'y': [4.57436224]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(2)':{1011.4942528768: {'x': [4.3509195], 'y': [3.30772468]}, 1388.2030178370899: {'x': [1.85004878], 'y': [3.42759165]}, 2026.7157991773738: {'x': [3.19392369], 'y': [2.20289567]}, 4028.3409449956025: {'x': [2.97054878], 'y': [4.61392989]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(3)':{1011.4942528768: {'x': [4.30008813], 'y': [3.33275688]}, 1385.826771657978: {'x': [1.93877343], 'y': [3.48216614]}, 2009.2898315582781: {'x': [3.16389162], 'y': [2.21530028]}, 4056.7375886654686: {'x': [3.15532032], 'y': [4.54476019]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(1)':{1007.8833853966516: {'x': [4.12204789], 'y': [3.16182994]}, 1383.6477987465664: {'x': [1.79012568], 'y': [3.2343386]}, 2008.5557026469878: {'x': [2.97468154], 'y': [1.85908656]}, 4056.737588665462: {'x': [2.89513616], 'y': [4.20441891]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(2)':{1009.560229448736: {'x': [4.15601649], 'y': [3.10099567]}, 1378.590078333393: {'x': [1.71850039], 'y': [3.22991874]}, 2013.071895431278: {'x': [2.92981559], 'y': [2.06343156]}, 4050.209205033878: {'x': [2.93370514], 'y': [4.28288237]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(3)diyigedengpailiangbian':{1007.6335877894846: {'x': [4.03472442], 'y': [3.13285742]}, 1396.8253968298668: {'x': [1.85338941], 'y': [3.0967188]}, 2030.7692307757268: {'x': [2.95915242], 'y': [1.91381182]}, 4033.333333346235: {'x': [2.9050642], 'y': [4.23795784]}}
        }



        lights_index, trajectory_index = trajectory_bag.my_align_time_index(image_time[min(image_time,key=image_time.get)],trajectory_time)

        fig.add_subplot(3,2,subIndex)
        subIndex+=1
        plot_all_light_map(experiment_folder,map_img,map_info,trajectory,trajectory_index,lights_world[experiment_folder],lights_real,multiple=True)
    plt.show()

    
