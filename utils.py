import numpy as np


# 2D bicubic interpolation is defined from 16 known nodes and required x and y
# idea taken from https://www.paulinternet.nl/?page=bicubic
def cubicInterpolation(p0, p1, p2, p3, x):
    return (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3)*(x**3)\
           + (p0 - 2.5*p1 + 2*p2 - 0.5*p3)*(x**2)\
           + (-0.5*p0 + 0.5*p2)*x + p1
def bicubicInterpolation(p00, p01, p02, p03, p10, p11, p12, p13,
                         p20, p21, p22, p23, p30, p31, p32, p33, x, y):
    return cubicInterpolation(cubicInterpolation(p00, p01, p02, p03, y),
                              cubicInterpolation(p10, p11, p12, p13, y),
                              cubicInterpolation(p20, p21, p22, p23, y),
                              cubicInterpolation(p30, p31, p32, p33, y), x)


# Definition of spline function from geo-spatial data analysis course
def Spline3(t):
    if abs(t) <= 1:
        return (1/6) * (((2 - abs(t))**3) - (4*((1 - abs(t))**3)))
    elif abs(t) <= 2:
        return (1/6) * (((2 - abs(t))**3))
    return 0

# definition of tensor product bicubic b-spline interpolation
# https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v05/undervisningsmateriale/kap7.pdf
def bicubicSpline(p00, p01, p02, p03, p10, p11, p12, p13,
                  p20, p21, p22, p23, p30, p31, p32, p33, x, y):
    b_x = []
    b_y = []
    for i in range(4):
        b_x.append(Spline3(x + 1 - i))
        b_y.append(Spline3(y + 1 - i))

    known_points = np.array([[p00, p01, p02, p03], [p10, p11, p12, p13],
                             [p20, p21, p22, p23], [p30, p31, p32, p33]])
    outer_matrix = np.outer(np.array(b_x), np.array(b_y))
    outer_matrix = np.moveaxis(np.tile(outer_matrix,(3,1,1)), [0,1,2], [2,0,1])

    return np.sum(np.sum(outer_matrix*known_points, axis=0),axis=0)