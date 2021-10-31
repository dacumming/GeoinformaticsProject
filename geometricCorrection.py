import matplotlib.pyplot as plt
from matplotlib import gridspec
import spectral.io.envi as envi
from osgeo import gdal
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

ctrl_pts_file='controlPoints.txt'
master_img_file='masterImageIndexes.txt'

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

# master image bands selection for plotting
master_img=master.read_bands([0,1,2])

# slave RGB image (bands selection for plotting)
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

# get georeferenced bounding box from image
ds = gdal.Open("dati/Avcilar_SPOTXS.img")
bounding_box=ds.GetGeoTransform()

# coordinates of top left coordinates of top left pixel
x_crs = bounding_box[0]
y_crs = bounding_box[3]

# pixel sizes
dx=bounding_box[1]
dy=bounding_box[5]

# master image dimensions
m_shp=master_img.shape

# this function takes the selected points and place each one of them into the
# correspondent array for slave and master images
def processControlPts(xcoords,ycoords):
    global x_crs
    global y_crs
    global dx
    global dy
    global m_shp

    x_master=[]
    y_master=[]
    x_slave=[]
    y_slave=[]
    for i in range(len(xcoords)):
        if i%2==0:
            x_master.append(x_crs+dx/2+float(xcoords[i])*dx)
            y_master.append(y_crs+dy/2+float(ycoords[i])*dy)
        else:
            x_slave.append(xcoords[i])
            y_slave.append(ycoords[i])

    return np.float64(x_master),\
           np.float64(y_master),\
           np.float64(x_slave),\
           np.float64(y_slave)


x_master, y_master, x_slave, y_slave = processControlPts(xcoords,ycoords)


max_degree=min(len(x_master)-1,10) # maximum polynomial degree
alphas=np.geomspace(0.0001,.1,20) # regularization values
degrees=np.arange(1,max_degree+1) #polynomial degrees values
seed=123 #random seed for lasso regression

# RMSE for each one of the algorithms
rmse_ridge=np.zeros((len(degrees),len(alphas)))
rmse_lasso=np.zeros((len(degrees),len(alphas)))
rmse_lr=np.zeros((len(degrees),1))
for a in range(len(alphas)):
    for d in range(len(degrees)):
        if a==0:# linear regression is computed just for 1 alpha
            se_lr = 0
        se_ridge = 0
        se_lasso = 0
        for i in range(len(x_master)):
            # definition of data sets for LOO CV
            x_train_x = np.delete(np.array([x_master, y_master]).T, i, 0)
            x_train_y = x_train_x
            y_train_x = np.delete(x_slave, i)
            y_train_y = np.delete(y_slave, i)
            x_val_x = np.array([x_master, y_master])[:, i]
            x_val_y = x_val_x
            y_val_x = x_slave[i]
            y_val_y = y_slave[i]

            # definition of polynomial features
            polynomial_x = PolynomialFeatures(degree=degrees[d])
            polynomial_y = PolynomialFeatures(degree=degrees[d])
            Xtr_x = polynomial_x.fit_transform(x_train_x)
            Xtr_y = polynomial_y.fit_transform(x_train_y)

            # linear regression model building for x and y. then square error is collected for each CV
            if a==0:
                lr_x = LinearRegression()
                lr_y = LinearRegression()
                lr_x.fit(Xtr_x, y_train_x)
                lr_y.fit(Xtr_y, y_train_y)
                Xp_x = polynomial_x.fit_transform(x_val_x.reshape(1, -1))
                Xp_y = polynomial_y.fit_transform(x_val_y.reshape(1, -1))
                yp_x = lr_x.predict(Xp_x)[0]
                yp_y = lr_y.predict(Xp_y)[0]
                se_lr += ((yp_x - y_val_x) ** 2) + ((yp_y - y_val_y) ** 2)

            # ridge model building for x and y. then square error is collected for each CV
            ridge_x = make_pipeline(StandardScaler(), Ridge(alpha=alphas[a]))
            ridge_y = make_pipeline(StandardScaler(), Ridge(alpha=alphas[a]))
            ridge_x.fit(Xtr_x, y_train_x)
            ridge_y.fit(Xtr_y, y_train_y)
            Xp_x = polynomial_x.fit_transform(x_val_x.reshape(1,-1))
            Xp_y = polynomial_y.fit_transform(x_val_y.reshape(1,-1))
            yp_x = ridge_x.predict(Xp_x)[0]
            yp_y = ridge_y.predict(Xp_y)[0]
            se_ridge+=((yp_x-y_val_x)**2)+((yp_y-y_val_y)**2)

            # lasso model building for x and y. then square error is collected for each CV
            lasso_x = make_pipeline(StandardScaler(), Lasso(alpha=alphas[a], max_iter=300, random_state=seed))
            lasso_y = make_pipeline(StandardScaler(), Lasso(alpha=alphas[a], max_iter=300, random_state=seed))
            lasso_x.fit(Xtr_x, y_train_x)
            lasso_y.fit(Xtr_y, y_train_y)
            Xp_x = polynomial_x.fit_transform(x_val_x.reshape(1, -1))
            Xp_y = polynomial_y.fit_transform(x_val_y.reshape(1, -1))
            yp_x = lasso_x.predict(Xp_x)[0]
            yp_y = lasso_y.predict(Xp_y)[0]
            se_lasso += ((yp_x - y_val_x) ** 2) + ((yp_y - y_val_y) ** 2)

        if a==0:
            rmse_lr[d] = np.sqrt(se_lr / len(x_master))
        rmse_ridge[d, a] = np.sqrt(se_ridge / len(x_master))
        rmse_lasso[d, a] = np.sqrt(se_lasso / len(x_master))

# all RMSEs are joined
rmse=np.concatenate([rmse_lr,rmse_ridge,rmse_lasso],axis=1)

# the best solution is stored to be shown in the figure
min_index=np.where(rmse == rmse.min())
if min_index[1][0]<1:
    best_model = 'Linear Regression'
    best_alpha = np.nan
    model_x = LinearRegression()
    model_y = LinearRegression()
elif min_index[1][0]<len(alphas)+1:
    best_model = 'Ridge Regression'
    best_alpha = alphas[min_index[1][0] - 1]
    model_x = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    model_y = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
else:
    best_model = 'Lasso Regression'
    best_alpha = alphas[min_index[1][0] - 21]
    model_x = make_pipeline(StandardScaler(), Lasso(alpha=best_alpha, max_iter=300, random_state=seed))
    model_y = make_pipeline(StandardScaler(), Lasso(alpha=best_alpha, max_iter=300, random_state=seed))

best_degree=degrees[min_index[0][0]]

# meshgrid of the parameters domain is defined
aa,dd=np.meshgrid(alphas,degrees)

# definition of the whole figure and the grid specification
fig = plt.figure(figsize=(14,7))
fig.suptitle('Leave-one-out cross-validation',size=15)
gs = gridspec.GridSpec(4, 9)

# the subplot for each model is built
ax = fig.add_subplot(gs[:3,:3])
ax.plot(degrees,rmse_lr)
ax.set_xlabel('polynomial degree')
ax.set_ylabel('RMSE')
ax.set_title('Linear Regression')
ax.grid()

ax = fig.add_subplot(gs[:3,3:6], projection='3d')
ax.plot_surface(aa, dd, rmse_ridge,cmap='jet', edgecolor='none')
ax.set_xlabel('Regularization')
ax.set_ylabel('polynomial degree')
ax.set_zlabel('RMSE')
ax.set_title('Ridge Regression')


ax = fig.add_subplot(gs[:3,6:9], projection='3d')
ax.plot_surface(aa, dd, rmse_lasso,cmap='jet', edgecolor='none')
ax.set_xlabel('Regularization')
ax.set_ylabel('polynomial degree')
ax.set_zlabel('RMSE')
ax.set_title('Lasso Regression')

# best result details displaying
fig.text(0.41,0.05,'Best model: '+best_model+'\n'+'Best RMSE: '+str(round(rmse.min(),2))+'\n'+
         'Best poly degree: '+str(best_degree)+'\n'+
         'Best regularization param: '+str(round(best_alpha,2)),size=14)
plt.show()

# the model is trained again with the best parameters and all the control points
x_train_x = np.array([x_master, y_master]).T
x_train_y = x_train_x
y_train_x = x_slave
y_train_y = y_slave
polynomial_x = PolynomialFeatures(degree=best_degree)
polynomial_y = PolynomialFeatures(degree=best_degree)
Xtr_x = polynomial_x.fit_transform(x_train_x)
Xtr_y = polynomial_y.fit_transform(x_train_y)
model_x.fit(Xtr_x, y_train_x)
model_y.fit(Xtr_y, y_train_y)

# all the pixels of the master image are mapped into the slave image indexing
x_img=np.arange(0,master_img.shape[1])*dx+dx/2+x_crs
y_img=np.arange(0,master_img.shape[0])*dy+dy/2+y_crs
X_img,Y_img=np.meshgrid(x_img,y_img)
xy_img=np.array([X_img.flatten('F'), Y_img.flatten('F')]).T #a.reshape(2, -1).T
Xp_x = polynomial_x.fit_transform(xy_img)
Xp_y = polynomial_y.fit_transform(xy_img)
yp_i = model_x.predict(Xp_x)
yp_j = model_y.predict(Xp_y)

# both index sets are plotted in the slave image indexes domain
I_original,J_original=np.meshgrid(np.arange(0,slave_img.shape[1]),np.arange(0,slave_img.shape[0]))
plt.figure(figsize=(9,9))
plt.scatter(I_original,J_original,s=0.01,label='Slave image')
plt.scatter(yp_i,yp_j,s=0.01,label='Master image')
plt.title('Geometric correction in the slave image indexes domain')
lgnd=plt.legend(loc="lower right")
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel('I')
plt.ylabel('J')
plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

# the mapping of the master image in the slave image indexing is stored
with open(master_img_file, 'w') as file:
    for i in range(len(yp_i)):
        file.writelines(str(yp_i[i]) + ' ' + str(yp_j[i]) + '\n')