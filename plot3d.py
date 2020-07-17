import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
vertexFile = open("./vertexPoints.txt", "r")

points = vertexFile.read().replace('[[', '').replace(']]', '').split('],[')

xdata = []
ydata = []
zdata = []


for point in points:
  stringData = point.replace('[', '')
  stringData = stringData.replace(']', '')
  coords = stringData.split(',')

  xdata.append(float(coords[0]))
  ydata.append(float(coords[1]))
  zdata.append(float(coords[2]))


xdata = np.array(xdata)
ydata = np.array(ydata)
zdata = np.array(zdata)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
plt.show()
