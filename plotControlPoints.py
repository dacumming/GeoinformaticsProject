import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spectral.io.envi as envi
import random

ctrl_pts_file='controlPoints.txt'

# read master image
header_master = 'dati/Avcilar_SPOTXS.hdr'
spectral_master = 'dati/Avcilar_SPOTXS.img'
master = envi.open(header_master, spectral_master)

# read slave image
header_slave = 'dati/Avcilar_landsat93.hdr'
spectral_slave = 'dati/Avcilar_landsat93'
slave = envi.open(header_slave, spectral_slave)

# all channels of slave
img = slave.read_bands(range(slave.shape[2]))

# master image
master_img=master.read_bands([0,1,2])

# slave RGB image
slave_img=slave.read_bands([3,2,1])

xcoords = []
ycoords = []

# read stored control points
with open(ctrl_pts_file,'r') as file:
    data = file.readlines()
    for line in data:
        coords = line.split()
        xcoords.append(coords[0])
        ycoords.append(coords[1])

# the following function plots a control point in an axis ax
def plotMarker(ax,img,x,y,color):
    ax.scatter(x, y,marker='+', s=100, color=color)
    ax.set_xlim([0, img.shape[1]-1])
    ax.set_ylim([0, img.shape[0]-1])
    ax.invert_yaxis()
    ax.figure.canvas.draw()

# plot master and slave images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,13))
ax1.imshow(master_img, picker=True)
ax2.imshow(slave_img, picker=True)

# load colors for control points plotting
nums = [random.randint(0, len(list(mcolors.TABLEAU_COLORS.values()))-1) for i in range(50)]
cols = [list(mcolors.TABLEAU_COLORS.values())[i] for i in nums]
count=0

# plot title
fig.suptitle('Control Points', fontsize="x-large", y=0.87)

# this loop plot each one of the control point in the correspondent axis
for i in range(len(xcoords)):
    if i%2==0:
        plotMarker(ax1, master_img, float(xcoords[i]), float(ycoords[i]), cols[count])
    else:
        plotMarker(ax2, slave_img, float(xcoords[i]), float(ycoords[i]), cols[count])
        count+=1

# show the plot and wait for pressing a button to continue
plt.show()
