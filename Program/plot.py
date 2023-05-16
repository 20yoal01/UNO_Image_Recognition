from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

blue = np.loadtxt('new_blue_bgr.csv', delimiter=",", dtype=int)
green = np.loadtxt('new_green_bgr.csv', delimiter=",", dtype=int)
red = np.loadtxt('new_red_bgr.csv', delimiter=",", dtype=int)
wild = np.loadtxt('new_wild_bgr.csv', delimiter=",", dtype=int)
yellow = np.loadtxt('new_yellow_bgr.csv', delimiter=",", dtype=int)
blue2 = np.loadtxt('blue_bgr.csv', delimiter=",", dtype=int)
green2 = np.loadtxt('green_bgr.csv', delimiter=",", dtype=int)
red2 = np.loadtxt('red_bgr.csv', delimiter=",", dtype=int)
wild2 = np.loadtxt('wild_bgr.csv', delimiter=",", dtype=int)
yellow2 = np.loadtxt('yellow_bgr.csv', delimiter=",", dtype=int)


print(blue[:,0])
 

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(blue[:,0], blue[:,1], blue[:,2], color = "blue")
ax.scatter3D(green[:,0], green[:,1], green[:,2], color = "green")
ax.scatter3D(red[:,0], red[:,1], red[:,2], color = "red")
ax.scatter3D(wild[:,0], wild[:,1], wild[:,2], color = "black")
ax.scatter3D(yellow[:,0], yellow[:,1], yellow[:,2], color = "yellow")
ax.scatter3D(blue2[:,0], blue2[:,1], blue2[:,2], color = "blue")
ax.scatter3D(green2[:,0], green2[:,1], green2[:,2], color = "green")
ax.scatter3D(red2[:,0], red2[:,1], red2[:,2], color = "red")
ax.scatter3D(wild2[:,0], wild2[:,1], wild2[:,2], color = "black")
ax.scatter3D(yellow2[:,0], yellow2[:,1], yellow2[:,2], color = "yellow")
ax.scatter3D(103, 134, 137, color="magenta")
ax.scatter3D([0,0,0,255,0,255,255,255],[0,0,255,0,255,255,255,0],[0,255,0,0,255,0,255,255], color="white" )

ax.set_xlabel('blue', fontweight ='bold')
ax.set_ylabel('green', fontweight ='bold')
ax.set_zlabel('red', fontweight ='bold')
plt.title("simple 3D scatter plot")
 

plt.show()