import time
start_time = time.time()
import matplotlib.pyplot as plt
from matplotlib import gridspec
import spectral.io.envi as envi
from osgeo import gdal
from sklearn.neighbors import KNeighborsRegressor
from utils import *
import warnings
warnings.filterwarnings("ignore")

# master image corrected indexes file name
master_img_file='masterImageIndexes.txt'

# name of the output resampled image
# supported formats: https://gdal.org/drivers/raster/index.html
resampled_img='Avcilar_resampled.tif'

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

# load stored geometrically corrected indexes of master image
i_master = []
j_master = []
with open(master_img_file,'r') as file:
    data = file.readlines()
    for line in data:
        indexes = line.split()
        i_master.append(indexes[0])
        j_master.append(indexes[1])
i_master = np.float64(i_master)
j_master = np.float64(j_master)
ij_master=np.array([i_master, j_master]).T

# get georeferenced bounding box from image
ds = gdal.Open("dati/Avcilar_SPOTXS.img")
bounding_box=ds.GetGeoTransform()

# coordinates of top left coordinates of top left pixel
x_crs = bounding_box[0]
y_crs = bounding_box[3]

# pixel sizes
dx=bounding_box[1]
dy=bounding_box[5]

# master and slave image dimensions
m_shp=master_img.shape
s_shp=slave_img.shape

# master image indexes shaped as 2D array
I_master=i_master.reshape(m_shp[0],m_shp[1],order='F')
J_master=j_master.reshape(m_shp[0],m_shp[1],order='F')

# a meshgrid of the slave image indexes is created, considering the boundaries of the master image
# a padding is added to perform correct interpolation at the boundaries of the master image
pad=10
I_slave, J_slave = np.meshgrid(np.arange(np.floor(min(min(i_master)-pad,0)),
                                         np.ceil(max(max(i_master)+pad,s_shp[1]))).astype(int),
                               np.arange(np.floor(min(min(j_master)-pad,0)),
                                         np.ceil(max(max(j_master)+pad,s_shp[0]))).astype(int))

# flatten the meshgrid for model fitting purposes
i_slave = I_slave.flatten('F')
j_slave = J_slave.flatten('F')
ij_slave=np.array([i_slave, j_slave]).T

# the slave image bands are stored in the padded version
# in this case the padding takes the band value in the nearest edge
bands_slave=np.zeros((I_slave.shape[0], I_slave.shape[1], slave.shape[2]))
for i in range(I_slave.shape[0]):
    for j in range(I_slave.shape[1]):
        if I_slave[i, j]<s_shp[1] and J_slave[i, j]<s_shp[0] and I_slave[i, j]>=0 and J_slave[i, j]>=0:
            bands_slave[i, j, :] = img[J_slave[i, j], I_slave[i, j], :]
        elif I_slave[i, j]>=s_shp[1]:
            bands_slave[i, j, :] = img[J_slave[i, j], s_shp[1] - 1, :]
        elif I_slave[i, j]<0:
            bands_slave[i, j, :] = img[J_slave[i, j], 0, :]
        elif J_slave[i, j] >= s_shp[0]:
            bands_slave[i, j, :] = img[s_shp[0] - 1, J_slave[i, j], :]
        elif J_slave[i, j] < 0:
            bands_slave[i, j, :] = img[0, J_slave[i, j], :]

# K neighbors regressor fitting and prediction for 3 different values of k
# the fitting is done over slave image indexes and bands
# the prediction is done over the master image indexes so the slave image is resampled
neigh_1 = KNeighborsRegressor(n_neighbors=1)
neigh_1.fit(ij_slave, bands_slave[:,:,[3,2,1]].reshape(len(ij_slave), 3,order='F'))
neigh_1_img=np.uint8(np.round(neigh_1.predict(ij_master).reshape(
    m_shp[0],m_shp[1],3,order='F')))

neigh_2 = KNeighborsRegressor(n_neighbors=2)
neigh_2.fit(ij_slave, bands_slave[:,:,[3,2,1]].reshape(len(ij_slave), 3,order='F'))
neigh_2_img=np.uint8(np.round(neigh_2.predict(ij_master).reshape(
    m_shp[0],m_shp[1],3,order='F')))

neigh_3 = KNeighborsRegressor(n_neighbors=3)
neigh_3.fit(ij_slave, bands_slave[:,:,[3,2,1]].reshape(len(ij_slave), 3,order='F'))
neigh_3_img=np.uint8(np.round(neigh_3.predict(ij_master).reshape(
    m_shp[0],m_shp[1],3,order='F')))

# definition of resampling images for bilinear, bicubic and cubic spline interpolations
linear_img = cubic_img = spline_img = np.zeros((m_shp[0],m_shp[1],3))

# integer indexes at the nearest 16 nodes of each of the master image indexes
i0 = np.floor(I_master).astype(int) - 1
i1 = i0 + 1
i2 = i0 + 2
i3 = i0 + 3
j0 = np.floor(J_master).astype(int) - 1
j1 = j0 + 1
j2 = j0 + 2
j3 = j0 + 3

# weighting of known data points at the 4 corner of each master image points for bilinear interpolation
# the idea is taken from the following code:
# https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
w_ul = (i2 - I_master) * (j2 - J_master)
w_ur = (i2 - I_master) * (J_master - j1)
w_ll = (I_master - i1) * (j2 - J_master)
w_lr = (I_master - i1) * (J_master - j1)






# master image indexes that are contained within the boundaries of the slave image are identified
in_img = (I_master>=0) & (I_master<s_shp[1]) & (J_master>=0) & (J_master<s_shp[0])

for i,j in np.ndindex(in_img.shape):# loop on the indexes of the master image
    # for each index of the master image that is contained in the slave image boundaries,
    # bilinear, bicubic and cubic spline interpolation is computing using the slave image band values
    if in_img[i,j]:
        # the values of the slave bands are defined in the nearest 16 slave pixels of each master image pixel
        for k in range(4):
            for l in range(4):
                exec('p'+str(k)+str(l)+'=bands_slave[j'+str(k)+'[i, j], i'+str(k)+'[i, j],[3,2,1]]')

        #  bilinear interpolation is computed in each master image pixel
        linear_img[i, j, :] = w_ul[i,j]*p11 + w_ur[i,j]*p12 \
                              + w_ll[i,j]*p21 + w_lr[i,j]*p22

        #  bicubic interpolation is computed in each master image pixel
        cubic_img[i, j, :] = bicubicInterpolation(p00, p01, p02, p03, p10, p11, p12, p13,
                                                  p20, p21, p22, p23, p30, p31, p32, p33,
                                                  np.tile(J_master[i, j] - j1[i, j], 3),
                                                  np.tile(I_master[i, j] - i1[i, j], 3))

        #  bicubic spline interpolation is computed in each master image pixel
        spline_img[i, j, :] = bicubicSpline(p00, p01, p02, p03, p10, p11, p12, p13,
                                            p20, p21, p22, p23, p30, p31, p32, p33,
                                            J_master[i, j] - j1[i, j], I_master[i, j] - i1[i, j])
    else:
        # as in the k nearest neighbor models the interpolation was computed in the paddings, now they are deleted
        neigh_1_img[i,j,:] = neigh_2_img[i,j,:] = neigh_3_img[i,j,:] = np.array([0, 0, 0])

# suitable format for images (8-bit integers)
linear_img = np.uint8(np.round(linear_img))
cubic_img = np.uint8(np.round(cubic_img))
spline_img = np.uint8(np.round(spline_img))


# definition of the whole figure and the grid specification
fig = plt.figure(figsize=(19,9))
fig.suptitle('Resampling',size=15,y=1.0)
gs = gridspec.GridSpec(2, 4)

# the subplot for each model is built
ax = fig.add_subplot(gs[0,0])
ax.imshow(neigh_1_img)
ax.set_title('K nearest neighbors. K = 1')

ax = fig.add_subplot(gs[0,1])
ax.imshow(neigh_2_img)
ax.set_title('K nearest neighbors. K = 2')

ax = fig.add_subplot(gs[0,2])
ax.imshow(neigh_3_img)
ax.set_title('K nearest neighbors. K = 3')

ax = fig.add_subplot(gs[1,0])
ax.imshow(linear_img)
ax.set_title('Bilinear interpolation')

ax = fig.add_subplot(gs[1,1])
ax.imshow(cubic_img)
ax.set_title('Bicubic interpolation')

ax = fig.add_subplot(gs[1,2])
ax.imshow(spline_img)
ax.set_title('Bicubic Spline interpolation')

ax = fig.add_subplot(gs[:,3])
ax.imshow(slave_img)
ax.set_title('Original slave image')

plt.tight_layout()
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
