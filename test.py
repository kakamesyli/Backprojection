# usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from backproject import create_guass
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad


pixel_range = 100
def create_init_img(pixel_range):
    init_img = np.zeros((pixel_range,pixel_range))
    gauss_range = 10
    gauss_1 = create_guass(gauss_range)
    init_img[30:30+gauss_range, 50:50+gauss_range] = gauss_1
    
    return init_img

def cal_flatfield_pim(p):
    mean_pim = cal_mean(p)
    variance_pim = cal_variance(p)
    flat_pim = np.zeros(len(p))
    for i in range(len(p)):
        if variance_pim == 0:
            flat_pim[i] = 0
        else:
            flat_pim[i] = ( p[i] - mean_pim ) / np.sqrt(variance_pim)
    return flat_pim

def cal_mean(p):
    mean_matrix = 0
    for time in range(len(p)):
        mean_matrix += p[time] / len(p)
        if mean_matrix < 1e-4:
            mean_matrix = 0
    #print(mean_matrix)
    return mean_matrix

def cal_variance(p):
    vari = 0
    variance_matrix = np.zeros(len(p))
    mean_matrix = cal_mean(p)
    for time in range(len(p)):
        variance_matrix[time] = (p[time] - mean_matrix)**2 / len(p)
        if variance_matrix[time] < 1e-4:
            variance_matrix[time] = 0
        vari += variance_matrix[time]
    #print(mean_matrix,vari)
    return vari

def func1(x):
    f = np.sin(np.sin(x))
    return f
def integ(f, a, b):
    y, error = quad(f, a, b)
    print('y={:.3f}, error={:.3f}'.format(y, error))

if __name__ == "__main__":
    '''
    a = create_init_img(pixel_range)
    plt.imshow(a)
    plt.show()
    '''
    '''
    p = [26,33,65,28,34,55,25,44,50,36,26,37,43,62,35,38,45,32,28,34]
    p_flat = cal_flatfield_pim(p)
    print(p_flat)
    '''
    
    p1 = [[1,2,3],[1,2,3],[1,2,3]]
    p = np.array([26,33,65,28,34,55,25,44,50,36,26,37,43,62,35,38,45,32,28,34])
    p = p.reshape((4,-1))
    fig = plt.figure()
    fig.suptitle('ttt')
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #fig.suptitle('A tale of 2 subplots')
    ax.imshow(p)
    ax.set_title("test")
    ax.set_xlabel('xxx')
    ax.set_ylabel('yxxx')
    

    #plt.imshow(p1,cmap = 'rainbow')
    #ax = plt.gca()
    #ax.set_xticklabels('tes')
    
    '''
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)

    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('A tale of 2 subplots')

    ax1.plot()
    ax1.set_ylabel('Damped oscillation')

    ax2.plot()
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Undamped')
    '''
    
    '''
    fig = plt.figure(1)
    ax = Axes3D(fig)
    X,Y = np.mgrid[0:pixel_range:1,0:pixel_range:1]
    f = X**2 + Y
    ax.plot_surface(X, Y, f, cmap='coolwarm', linewidth=0, antialiased=False)
    '''
    # plt.show()

    w = 90/180*np.pi
    integ(func1, 0+w, 2*np.pi+w)