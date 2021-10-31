import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spectral.io.envi as envi
import random

ctrl_pts_file='file1.txt'

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

# the following function plots the selected point
def plotMarker(ax,img,x,y,color):
    ax.scatter(x, y,marker='+', s=100, color=color)
    ax.set_xlim([0, img.shape[1]-1])
    ax.set_ylim([0, img.shape[0]-1])
    ax.invert_yaxis()
    ax.figure.canvas.draw()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))

ax1.imshow(master_img, picker=True)
ax2.imshow(slave_img, picker=True)
nums = [random.randint(0, len(list(mcolors.TABLEAU_COLORS.values()))-1) for i in range(50)]
cols = [list(mcolors.TABLEAU_COLORS.values())[i] for i in nums]
count=0
fig.suptitle('Please select Control Points.', fontsize="x-large", y=0.87)
switch=True

def addPoints(x,y):
    global xcoords
    global ycoords
    xcoords.append(x)
    ycoords.append(y)

# function activated when an image is clicked
# it collects selected point coordinates and plot them over the image
# the point selection must be alternated among the master and slave images
def onPick(event):
    global switch
    global count
    if event.inaxes == ax1 and switch:
        addPoints(event.xdata,event.ydata)
        plotMarker(ax1,master_img,event.xdata,event.ydata,cols[count])
        switch = False
    elif event.inaxes == ax2 and not switch:
        addPoints(event.xdata, event.ydata)
        plotMarker(ax2,slave_img,event.xdata,event.ydata,cols[count])
        count+=1
        switch=True
    else:
        print('please select a point in the other image')

    with open(ctrl_pts_file, 'w') as file:
        for i in range(len(xcoords)):
            file.writelines(str(xcoords[i]) + ' ' + str(ycoords[i]) + '\n')

fig.canvas.mpl_connect('button_press_event', onPick)

plt.show()
