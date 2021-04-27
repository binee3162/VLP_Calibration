#coding=utf-8
#!/usr/bin/env python2
# @brief plot CDF graph and boxplot of calibration errors exported from map_lights.py
# @author Song Gao

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc
import pandas as pd
lights_world={
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(1)second diergepinlvcuole':{1000: {'x': [4.30477277], 'y': [3.26177099]}, 1375: {'x': [1.9275354], 'y': [3.51809512]}, 2000: {'x': [3.1969149], 'y': [2.26302395]}, 4000: {'x': [3.18190249], 'y': [4.57436224]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(2)':{1000: {'x': [4.3509195], 'y': [3.30772468]}, 1375: {'x': [1.85004878], 'y': [3.42759165]}, 2000: {'x': [3.19392369], 'y': [2.20289567]}, 4000: {'x': [2.97054878], 'y': [4.61392989]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421051636/210421051636(3)':{1000: {'x': [4.30008813], 'y': [3.33275688]}, 1375: {'x': [1.93877343], 'y': [3.48216614]}, 2000: {'x': [3.16389162], 'y': [2.21530028]}, 4000: {'x': [3.15532032], 'y': [4.54476019]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(1)':{1000: {'x': [4.12204789], 'y': [3.16182994]}, 1375: {'x': [1.79012568], 'y': [3.2343386]}, 2000: {'x': [2.97468154], 'y': [1.85908656]},4000: {'x': [2.89513616], 'y': [4.20441891]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(2)':{1000: {'x': [4.15601649], 'y': [3.10099567]}, 1375: {'x': [1.71850039], 'y': [3.22991874]}, 2000: {'x': [2.92981559], 'y': [2.06343156]},4000: {'x': [2.93370514], 'y': [4.28288237]}},
            '/home/song/Desktop/catkin_ws/src/vlp_camera/calibrationBag/data/210421061741/210421061741(3)diyigedengpailiangbian':{1000: {'x': [4.03472442], 'y': [3.13285742]}, 1375: {'x': [1.85338941], 'y': [3.0967188]}, 2000: {'x': [2.95915242], 'y': [1.91381182]},4000: {'x': [2.9050642], 'y': [4.23795784]}}
        }
div=[]
for key in lights_world.keys():
    if key.split("/")[-2]=='210421061741':
        lights_real={'1000':[4.13, 3.07],'2000':[2.98, 1.94],'1375':[1.77, 3.17],'4000':[2.93, 4.30]}  #for the last package
    else:
        lights_real={'1000':[4.33, 3.39], '2000':[3.15, 2.28],'1375': [1.94, 3.57],'4000': [3.12, 4.68]} #last but one
    for j in lights_world[key].keys():
        real=lights_real[str(j)]
        d=np.sqrt((lights_world[key][j]['x'][0]-real[0])**2+(lights_world[key][j]['y'][0]-real[1])**2)
        div=np.append(div,d)

dataframe = pd.DataFrame(div)

plt.figure()
plt.boxplot(div,showmeans=True)

plt.ylabel("Error[m]")
plt.title("Boxplot of calibration errors")
plt.show()
l=len(div)
div, div_counts = np.unique(div, return_counts=True)
div_counts=div_counts/float(l)
cdf=np.cumsum(div_counts)
plt.step(div,cdf)


plt.xlabel("Error(m)")
plt.ylabel("Probability")
plt.title("CDF for distance errors")
plt.show()
