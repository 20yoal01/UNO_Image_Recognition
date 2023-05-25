from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

PATH = 'KNN generator/test/'

blue = np.loadtxt(PATH + 'new_blue_bgr.csv', delimiter=",", dtype=int)
green = np.loadtxt(PATH + 'new_green_bgr.csv', delimiter=",", dtype=int)
red = np.loadtxt(PATH + 'new_red_bgr.csv', delimiter=",", dtype=int)
#wild = np.loadtxt(PATH + 'wild_bgr.csv', delimiter=",", dtype=int)
yellow = np.loadtxt(PATH + 'new_yellow_bgr.csv', delimiter=",", dtype=int)

print(blue[:,0])
 

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(blue[:,0], blue[:,1], blue[:,2], color = "blue")
ax.scatter3D(green[:,0], green[:,1], green[:,2], color = "green")
ax.scatter3D(red[:,0], red[:,1], red[:,2], color = "red")
#ax.scatter3D(wild[:,0], wild[:,1], wild[:,2], color = "black")
ax.scatter3D(yellow[:,0], yellow[:,1], yellow[:,2], color = "yellow")
ax.scatter3D([0,0,0,180,0,180,180,180],[0,0,255,0,255,255,255,0],[0,255,0,0,255,0,255,255], color="white" )
ax.scatter3D(103, 134, 137, color="magenta")

ax.set_xlabel('hue', fontweight ='bold')
ax.set_ylabel('saturation', fontweight ='bold')
ax.set_zlabel('value', fontweight ='bold')
plt.title("KNN dataset")
 

plt.show()