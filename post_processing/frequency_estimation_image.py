#coding=utf-8
#!/usr/bin/env python2
# @brief Estimate the frequency of a light source in all images in a folder and plot hist graph
# @author Song Gao
# @bug No known bugs

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import vlp_classes as vlp
#import statsmodels.api as sm



# Get all images in folder
#root_folder = "../../bag/test_waveform/"
root_folder = "/mnt/c/Users/Song/Desktop/master/results/0419"
image_paths = glob.glob(root_folder + "/*/*.jpg") 

camera = vlp.calibration()

frequencies_dict = {"1000":[],"1375":[],"1700":[],"2000":[],"2500":[],"3000":[],"4000":[]}
frequencies={}
centroids = {}
averages={}
width = 5

for image_path in image_paths:
    trueFreq=image_path.split("/")[-2]
    image = cv2.imread(image_path)
    im2, coordinate_l, radius_l = camera.img_preprocess(image) 

    for index in range(len(coordinate_l)):  

        if camera.check_image_center(coordinate_l[index]):   
            light_frequency, average = camera.my_get_frequency(image, coordinate_l[index],radius_l[index])
            frequencies[image_path] = light_frequency
            if light_frequency!=0:
                frequencies_dict[trueFreq]=np.append(frequencies_dict[trueFreq],light_frequency)
            averages[image_path]=average

fig,axs = plt.subplots(2,4,figsize=(12,6))
def plot_pdf_hist(data,axis,color,label):
    axis.hist(data,20,density=1,histtype='bar',facecolor=color,alpha=0.75,label=label,rwidth=5)
    axis.set_xlabel('frequency(Hz)',{'size':10})
    axis.set_ylabel('probability',{'size':12})
    axis.tick_params(labelsize=12)

plot_pdf_hist(frequencies_dict["4000"],axs[1,2],'red','4000')
plot_pdf_hist(frequencies_dict["2000"],axs[0,3],'green','2000')
plot_pdf_hist(frequencies_dict["1000"][0:20],axs[0,0],'blue','1000')
plot_pdf_hist(frequencies_dict["1375"][0:20],axs[0,1],'yellow','1375')
plot_pdf_hist(frequencies_dict["2500"],axs[1,0],'purple','2500')
plot_pdf_hist(frequencies_dict["3000"],axs[1,1],'orange','3000')
plot_pdf_hist(frequencies_dict["1700"],axs[0,2],'cyan','1700')
axs[1,3].remove()
# plt.xlabel('frequency(Hz)')
# plt.ylabel('probability')
# plt.legend(loc="upper left",prop={'size':12})
plt.show()
print frequencies
print centroids
