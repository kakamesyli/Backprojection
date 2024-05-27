#!usr/bin/env python
# -*- coding:utf-8 -*-

'''
Created on 20210530

@author: think
'''
import copy

import numpy as np
from cmath import pi
from copy import deepcopy
from _functools import reduce

import scipy.signal
from matplotlib import pyplot as plt
from numpy.random.mtrand import multivariate_normal
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import projection_registry




grid_dis = 1.2e3
dector_area = 10
component = 40
satellite_rot_speed = 1*pi/180
module_range = 180
time_intev = ((module_range/component)*pi/180)/satellite_rot_speed
init_phase_pim = 0
FOV = 30/(2*pi)
pixel_range = 128

class Base_oper(object):
    def assign(self,value):
        return value
    

class Source_para(Base_oper):
    def __init__(self, source_theta,source_phai,source_intense):
        self.source_theta = super().assign(source_theta)
        self.source_phai = super().assign(source_phai)
        self.source_intense = super().assign(source_intense)


class Pim(Source_para):
    def __init__(self, time_num, pitch, grid_angle, source_theta, source_phai, source_intense):
        self.time_num = super().assign(time_num)
        #self.pixel_num = super().assign(pixel_num)
        self.pitch = super().assign(pitch)
        self.grid_angle = super().assign(grid_angle)
        
        self.pim_value = 0
        
        #继承父类的参数
        super().__init__(source_theta, source_phai, source_intense)
        
    def cal_pim(self):
        K = (2*pi/self.pitch)
        c = grid_dis * np.tan(self.source_theta) * np.cos(self.source_phai + (self.time_num)*satellite_rot_speed*time_intev - self.grid_angle)
        self.pim_value = 0.25 + (2/pi**2)*np.cos(K*c - init_phase_pim)
        
    
    def calUVSpaceApproach(self):
        K = (2*pi/self.pitch)
        c = grid_dis * np.tan(self.source_theta) * np.cos(self.source_phai - self.grid_angle)
        UValue = 0.25 + (2/pi**2)*np.cos(K*c - init_phase_pim)
        VValue = 0.25 + (2/pi**2)*np.sin(K*c - init_phase_pim)
        return np.array([UValue,VValue])


def calUVTransform(pitch,grid_angle,source_theta,source_phai):
    K = (2*pi/pitch)
    c = grid_dis * np.tan(source_theta) * np.sin(source_phai - grid_angle)
    UTransform = 0.25+((2/pi**2)+2/(3*pi)**2)*np.cos(K*c - init_phase_pim)
    VTransform = 0.25+((2/pi**2)+2/(3*pi)**2)*np.sin(K*c - init_phase_pim)

    return np.array([UTransform, VTransform])

def calUVTransformCouple(pitch,grid_angle,source_theta,source_phai):
    K = (2*pi/pitch)
    c = grid_dis * np.tan(source_theta) * np.sin(source_phai - grid_angle)
    cCouple = grid_dis * np.tan(source_theta) * np.sin(source_phai - grid_angle -pi)
    UTransform = 0.25+((2/pi**2)+2/(3*pi)**2)*np.cos(K*c - init_phase_pim)
    VTransform = 0.25+((2/pi**2)+2/(3*pi)**2)*np.sin(K*c - init_phase_pim)
    UTransformCouple = 0.25+((2/pi**2)+2/(3*pi)**2)*np.cos(K*cCouple - init_phase_pim)
    VTransformCouple = 0.25+((2/pi**2)+2/(3*pi)**2)*np.sin(K*c - init_phase_pim)#sin不能变
    
    return np.array([UTransform, VTransform, UTransformCouple, VTransformCouple])

class Ci(Pim):
    def __init__(self, time_num, img_matrix, pitch, grid_angle):
        self.pim_matrix = []
        self.cim_value_matrix = []
        self.ci_value = 0
        
        for i in range(len(img_matrix)):
            for j in range(len(img_matrix)):
                p_pos = pixel_position(img_matrix, j, i)
                self.pim_matrix.append(Pim(time_num, pitch, grid_angle, p_pos[0,0], p_pos[0,1], img_matrix[i][j]))

            '''
            self.pi_matrix.append(self.make_pi(time_num, i, pos_r, pos_phai, pitch, alpha))
            pim_value = self.make_pi(time_num, i, pos_r, pos_phai, pitch, alpha).pim_value
            cal_cim(pim_value, source_intense)
            self.cim.append(pim_value * source_intense)
        '''
        
    #计算出图像的pim
    def set_pim_value(self):
        p_value = []
        for pim in self.pim_matrix:
            pim.cal_pim()
            p_value.append(pim.pim_value)
        p_value_array = np.array(deepcopy(p_value, memo=None, _nil=[]))
        p_value_array = p_value_array.reshape(pixel_range,pixel_range)
        return p_value_array
    
    #计算pim*t的求和，返回的是像元分布矩阵
    def pim_time(self):
        pim_time = []
        self.set_pim_value()
        for pim in self.pim_matrix:
            pim_time.append(pim.pim_value * time_intev)
        return pim_time
          
    def set_cim_value(self):
        self.set_pim_value()
        for pim in self.pim_matrix:
            self.cim_value_matrix.append(pim.pim_value * pim.source_intense * time_intev)
        return self.cim_value_matrix
    
    #计算所有像元在时间仓i下的的ci值，形成矩阵[图像矩阵ci，求和ci值]ֵ
    def cal_ci_value(self):
        cim_value_matrix = self.set_cim_value()
        self.ci_value = reduce(lambda x,y:x+y,self.cim_value_matrix)
        return [cim_value_matrix, self.ci_value]
    
    #单个像元的ci值对所有时间仓i求和，求解源强度                                                                                                                                                                                                                                                                                                                                                                                                                                                             

'''     

    def make_pi(self, time_num, pixel_num, pos_r, pos_phai, pitch, alpha):
        pim = pim(time_num, pixel_num, pitch, alpha)
        pim.cal_pim(pos_r, pos_phai, pitch, alpha)
        return pim

    def cal_ci(self,pim_value,source_intense,pixel_num):
        for i in range(pixel_num):
            self.cim_value = cal_cim(pim_value,source_intense)
        


def cal_pim(pos_r,pos_phai,pitch,angle):
    K = 2*PI/pitch
    c = pos_r * cmath.cos(pos_phai-angle)
    pim_value = 0.25 + (2/PI^2) * cmath.cos(K*c)
    return pim_value

def cal_cim(pim_value,source_intense):
    cim_value = pim_value * (dector_area*time_intev) * source_intense
    return cim_value

'''
def create_init_img(pixel_range):
    init_img = np.zeros((pixel_range,pixel_range))
    guass_image1 = np.zeros_like(init_img)
    guass_image2 = np.zeros_like(init_img)
    
    '''
    init_img[7,9] = 14872
    init_img[5,10] = 14872
    init_img[5,11] = 14872
    '''
    #init_img[int(np.floor(pixel_range/1.6)),int(np.floor(pixel_range/1.4))] = 1e3
    #init_img[int(np.floor(pixel_range/3.1)),int(np.floor(pixel_range/2.2))] = 1e3
    
    
    
    #高斯点源
    gauss_range_1 = 44
    gauss_range_2 = 20
    
    #directreverse
    '''
    gauss_pos_1 = [45,23]
    gauss_pos_2 = [11,41]
    '''
    
    gauss_pos_1 = [32,37]
    gauss_pos_2 = [37,42]
    sigma = 2
    gauss_1 = create_guass(gauss_range_1, sigma)
    gauss_2 = create_guass(gauss_range_2, sigma)
    guass_image1[gauss_pos_1[0]-int(gauss_range_1/2):gauss_pos_1[0]+int(gauss_range_1/2), gauss_pos_1[1]-int(gauss_range_1/2):gauss_pos_1[1]+int(gauss_range_1/2)] = gauss_1
    guass_image2[gauss_pos_2[0]-int(gauss_range_2/2):gauss_pos_2[0]+int(gauss_range_2/2), gauss_pos_2[1]-int(gauss_range_2/2):gauss_pos_2[1]+int(gauss_range_2/2)] = gauss_2*0.2
    init_img = init_img + guass_image1

    return init_img

def pixel_position(img_matrix,x,y):
    length = img_matrix.shape[1]
    coor_ = FOV/length
    x_angle = (x-(length-1)/2)*coor_
    y_angle = (y-(length-1)/2)*coor_
    theta = np.sqrt(x_angle**2+y_angle**2) / grid_dis
    if x_angle == 0:
        phai = pi/2
    else:
        phai = np.arctan(y_angle/x_angle)
    return np.array([theta,phai,img_matrix[y][x]])[np.newaxis,:]


def calPolarCoordinate(imgMatrix):#按行排序的极坐标
    ImagePolarCoordinate = np.array([])
    length = imgMatrix.shape[0]
    for y in range(length):
        for x in range(length):
            if ImagePolarCoordinate.any():
                ImagePolarCoordinate = np.concatenate((ImagePolarCoordinate,pixel_position(imgMatrix, x, y)),axis=0)
            else:
                ImagePolarCoordinate = pixel_position(imgMatrix, x, y)
    return ImagePolarCoordinate
    
'''
def ci_value_total_pixel(time_num, img_matrix, pitch, grid_angle):
    ci = Ci(time_num, img_matrix, pitch, grid_angle)
    return ci.cal_ci_value()
'''

def create_guass(p, Sigma):
    x,y = np.mgrid[-10:10:p*1j, -10:10:p*1j]
    sigma = Sigma
    z = 1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2))
    return z

    '''
    mu = (np.median(x), np.median(y))
    sigma = np.array([0.25, 0.25])
    covariance = np.diag(sigma**2)
    size = (p,p)
    z = multivariate_normal(mu,covariance,size)
    return z
    '''
    
    
def ci_value_total_pixel(ci):
    return ci.cal_ci_value()

def pi_value_total_pixel(ci):
    return ci.pim_time()
#求图像中的每个像元在时间仓i下的调制曲线值
def cal_pi_value(ci):
    return ci.set_pim_value()

def cim_value_matrix(time_num, img_matrix, pitch, grid_angle):
    ci = Ci(time_num, img_matrix, pitch, grid_angle)
    return ci.set_cim_value()

#某个像元位置产生的计数曲线
def plot_cim_pixel(x,y,pixel_range,matrix_in,component):
    cim = []
    for i in range(component):
        cim.append(matrix_in[i][x*pixel_range+y])
    ci = reduce(lambda x,y:x+y, cim)
    plt.plot(np.linspace(0,component-1,component), cim, linewidth = 0.5, color = 'r')
    return ci
    
def backproject(x,y,matrix_c_in, matrix_p_in, component):
    cim = []
    pim_time = []
    for i in range(component):
        cim.append(matrix_c_in[i][x*pixel_range+y])
        pim_time.append(matrix_p_in[i][x*pixel_range+y])
    ci_sum = reduce(lambda x,y:x+y, cim)
    pi_sum = reduce(lambda x,y:x+y, pim_time)
    if pi_sum == 0:
        return 0
    else:
        return ci_sum/pi_sum
def backprojection_base(p,c):
    f= np.zeros((pixel_range,pixel_range))
    for x in range(pixel_range):
        for y in range(pixel_range):
            f[x][y] = (p[x][y] * c / time_intev)
    return f
    
def backprojection(p,c):
    '''
    f = np.zeros((pixel_range,pixel_range))
    for x in range(pixel_range):
        for y in range(pixel_range):
            for i in range(component):
                f[x][y] += (p[i][x][y] * c[i] / time_intev)
    '''
    
    f = np.zeros((pixel_range,pixel_range))
    for i in range(component):
        f += backprojection_base(p[i], c[i])
    return f

'''
def backprojection_nflat(p,c):
    f = np.zeros((pixel_range,pixel_range))
    for x in range(pixel_range):
        for y in range(pixel_range):
            for i in range(component):
                f[x][y] += (p[i][x][y] * c[i] / time_intev  )
    return f
'''

def cal_flatfield_pim(p):
    mean_pim = cal_mean(p)
    variance_pim = cal_variance(p)
    flat_pim = np.zeros((component,pixel_range,pixel_range))
    for x in range(pixel_range):
        for y in range(pixel_range):
            for i in range(component):
                if variance_pim[x][y] == 0:#mean计算可能存在误差，导致样本点与均值相减时不为零；但是由于方差计算时，当方差过小时是设置为0，否则会出现过大数，因此可能出现除0的情况，但是由于方差为0，则样本点实际上值都相同，相当于样本值与均值相等，平场变化时相减为0，因此这里做条件约束
                    flat_pim[i][x][y] = 0
                else:
                    #flat_pim[i][x][y] = ( p[i][x*pixel_range+y] - mean_pim[x][y] ) / np.sqrt(variance_pim[x][y])
                    flat_pim[i][x][y] = ( p[i][x][y] - mean_pim[x][y] ) / np.sqrt(variance_pim[x][y])
                    # flat_pim[i][x][y] = (p[i][x][y] - mean_pim[x][y]) / variance_pim[x][y]
    return flat_pim

def cal_mean(p):
    mean_matrix = np.zeros((pixel_range, pixel_range))
    for x in range(pixel_range):
        for y in range(pixel_range):
            for time in range(component):
                mean_matrix[x][y] += p[time][x][y] / component
                if mean_matrix[x][y] < 1e-4:
                    mean_matrix[x][y] = 0
    return mean_matrix

def cal_variance(p):
    variance_matrix = np.zeros((pixel_range, pixel_range))
    mean_matrix = cal_mean(p)
    for x in range(pixel_range):
        for y in range(pixel_range):
            for time in range(component):
                variance_matrix[x][y] += (p[time][x][y] - mean_matrix[x][y])**2 / component
                # if variance_matrix[x][y] < 1e-4:
                #     variance_matrix[x][y] = 0
    return variance_matrix

def pulse_res(theta,a):
    l = 1200
    f = scipy.special.jv(0, 2*np.pi*l/a*np.tan(theta))
    return f

def synPSF(pitch_array, fov):
    y = np.array([])
    fig = plt.figure('eachPSF')
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlabel(r'$\theta\,/\,arcsec$', fontsize=10)
    ax.set_ylabel(r'$Amplitude$', fontsize=10)
    ax.set_xticks(np.linspace(-fov, fov, 10), np.int16(np.round(np.linspace(-fov, fov, 10)*57.3*3600, decimals=0)))
    save_path = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\不同空间分辨率狭缝的脉冲响应.pdf'
    theta = np.linspace(0, fov, 500)
    for p in range(len(pitch_array)):
        fpo = pulse_res(theta, pitch_array[p])
        fna = np.flip(fpo)
        f = np.concatenate((fna, fpo))
        if y.any():
            y = np.append(y, f[np.newaxis,:], axis=0)
        else:
            y = f[np.newaxis, :]
        ax.plot(np.linspace(-fov, fov, 1000),y[p,:], label = r'$Grid\,{}={:.2f}\,arcsec$'.format(p+1, pitch_array[p] / grid_dis * 57.3 * 3600/2), linewidth=0.7)
    # a = np.array([[8, 16, 24, 36, 15, -10, 5, -4, 3, -2]])
    # y1 = np.dot(a,y)
    syn_y = reduce(lambda a, b: a + b, y)
    # syn_y1 = reduce(lambda a, b: a + b, y1)
    # syn_y1 = reduce(lambda a, b: a + b, y[:5])
    # syn_y2 = reduce(lambda a, b: a + b, y[4:-1])
    ax.plot(np.linspace(-fov, fov, 1000), syn_y[:]/len(pitch_array), label=r'$synthesis\,PSF\,(1-10)$', ls = '-', linewidth=1.5, color='k')
    # ax.plot(np.linspace(-fov, fov, 1000), syn_y1[:], label=r'$synthesis\,PSF\,(1-5)$', ls = ':', linewidth=1.2)
    # ax.plot(np.linspace(-fov, fov, 1000), syn_y2[:], label=r'$synthesis\,PSF\,(5-9)$', ls = ':', linewidth=1.2)
    plt.legend(loc='upper right', ncol=1, bbox_to_anchor=(1,1), fontsize = 8, labelspacing=0.8,
           handlelength=0.5, handletextpad=0.5, handleheight=0.5, fancybox=True, edgecolor='k')
    # plt.savefig(save_path, dpi=1000, format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    init_img = create_init_img(pixel_range)

    #增加初始图像按照强度-透明度进行重叠
    init_img_alpha = copy.deepcopy(init_img)
    for i, v in enumerate(init_img_alpha):
        if v.any():
            for i1, v1 in enumerate(init_img_alpha[i,:]):
                if v1 > 1e-3:
                    init_img_alpha[i,i1] = 1
    # init_img_alpha = init_img * np.ones_like(init_img) / np.max(init_img)
    #显示原始图像
    
    
    # pitch_array = [36e-3,52e-3,76e-3,108e-3,156e-3,224e-3,344e-3,524e-3,800e-3,1224e-3]
    # pitch_array = [36e-3,48e-3,56e-3,76e-3,122e-3,176e-3,224e-3,344e-3,500e-3,668e-3]
    # pitch_array = [36e-3, 40e-3, 48e-3, 56e-3, 68e-3, 80e-3, 124e-3, 244e-3, 344e-3, 500e-3]
    pitch_array = [36e-3, 40e-3, 48e-3, 56e-3]
    # pitch_array = [1224e-3]
    grid_angle = 0
    ci = []
    cim = []
    #pim = []
    p_value = []
    p_flat = []#放置归一化的pim

    f = np.zeros((pixel_range, pixel_range))
    farray = np.array([])
    x_pixel = 64
    y_pixel = 64
    x_pixel1 = 32
    y_pixel1 = 37
    x_pixel2 = 4
    y_pixel2 = 4
    time_test = 0
    fig_module_curve = plt.figure('module curve')
    fig_trans_3d = plt.figure('trans 3d')
    fa1 = np.zeros((len(pitch_array),pixel_range,pixel_range))
    for p in range(len(pitch_array)):
        ci.append([])
        cim.append([])
        #pim.append([])
        p_value.append([])#注意需要先扩展一个空间，结构为 p_value[pitch][time][x][y]
        p_flat.append([])

        
        for time_num in range(component):
            c = Ci(time_num, init_img, pitch_array[p], grid_angle)
            [cim_out,ci_out] = ci_value_total_pixel(c)
            
            #[cim_out,ci_out] = ci_value_total_pixel(time_num, init_img, pitch_array[p], grid_angle)
            
            ci[p].append(ci_out)#所有像元在时间仓下的和
            cim[p].append(cim_out)#所有时间仓下的像元光子计数分布
            #pim[p].append(pi_value_total_pixel(c))#所有时间仓下的像元pim*t
            p_value[p].append(cal_pi_value(c))

            #ci[p].append(ci_value_total_pixel(time_num, init_img, pitch_array[p], grid_angle)[1])
            #cim[p].append(ci_value_total_pixel(time_num, init_img, pitch_array[p], grid_angle)[0])
            
        #计算平场变化pim
        p_flat[p] = cal_flatfield_pim(p_value[p])
        
        # p_mean = cal_mean(p_value[p])
        # fig = plt.figure()
        # ax_mean = fig.add_axes([0.05,0.05,0.9,0.9])
        # ax_mean.imshow(p_mean, cmap = 'gray')


        
        '''
        #显示pim图像，数字代表时间仓
        pim_flat_vol = np.array(deepcopy(p_flat[p][time_test], memo=None, _nil=[]))
        pim_flat_vol = pim_flat_vol.reshape(pixel_range,pixel_range)
        pim_vol = np.array(deepcopy(p_value[p][time_test], memo=None, _nil=[]))
        pim_vol = pim_vol.reshape(pixel_range,pixel_range)
        '''
        
        #显示单个时间仓下反投影图像
        f_flat = backprojection_base(p_flat[p][time_test], ci[p][time_test])
        f_nonflat = backprojection_base(p_value[p][time_test], ci[p][time_test])
        #显示图像像元强度与自身pim相乘的图像
        c_flat = np.multiply(init_img,p_flat[p])
        c_nonflat = np.multiply(init_img,p_value[p])
        

        # #将p_flat图像按一定时间仓绘图
        # time_bin = 24
        # fig = plt.figure()
        # for i in range(time_bin):
        #     ax_time = fig.add_subplot(4,6,i+1)
        #     #绘制非平场变化pim图像
        #     #ax_time.imshow(p_value[p][i], cmap = 'gray')
        #     #绘制平场变化pim图像
        #     ax_time.imshow(p_flat[p][i], cmap = 'gray')
        #     #fig.gca()
        #     ax_time.set_xticklabels([])
        #     ax_time.set_yticklabels([])

        
        '''
        fig = plt.figure()
        X,Y = np.mgrid[0:pixel_range:1,0:pixel_range:1]
        ax_pim = fig.add_axes([0.05,0.05,0.9,0.9], projection = '3d')
        ax_pim.plot_surface(X,Y,p_value[p][12],cmap = 'coolwarm')
        ax_pim.set_xlabel('x')
        ax_pim.set_ylabel('y')
        ax_pim.set_zlabel('z')
        ax_pim.set_zlim(-0.5,1)
        '''
            
        '''
                       #绘图部分
        fig = plt.figure()
        X,Y = np.mgrid[0:pixel_range:1,0:pixel_range:1]
        
        #ax1 = fig.add_axes([0.05,0.1,0.5,0.5])
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(f_flat,cmap = 'gray')
        ax1.set_title('flat')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        #ax2 = fig.add_axes([0.55,0.1,0.5,0.5])
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(f_nonflat,cmap = 'gray')
        ax2.set_title('non_flat')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        ax1_3d = fig.add_axes([0.05,0.05,0.5,0.5], projection = '3d')
        #ax1_3d = fig.add_subplot(2,2,3, projection = '3d')
        ax1_3d.plot_surface(X,Y,f_flat, cmap='coolwarm', linewidth=0, antialiased=False)
        ax1_3d.set_title('flat_3d')
        ax1_3d.set_xlabel('x_3d')
        ax1_3d.set_ylabel('y_3d')
        ax1_3d.set_zlabel('z_3d')
        
        ax2_3d = fig.add_axes([0.45,0.05,0.5,0.5], projection = '3d')
        #ax2_3d = fig.add_subplot(2,2,4, projection = '3d')
        ax2_3d.plot_surface(X,Y,f_nonflat, cmap='coolwarm', linewidth=0, antialiased=False)
        ax2_3d.set_title('nonflat_3d')
        ax2_3d.set_xlabel('x_3d')
        ax2_3d.set_ylabel('y_3d')
        ax2_3d.set_zlabel('z_3d')
        '''
        
        '''
        t = []
        for i in range(component):
            t.append(p_flat[p][i][12][12])
        plt.figure(2)
        plt.plot(np.linspace(0,component-1,component), t, linewidth = 0.5, color = 'r')
        '''


        #source_inte = backproject(x_pixel, y_pixel, cim[p], pim[p], component)
        
        '''
        #绘制整个图像的ci曲线
        plt.figure(3)
        plt.subplot(len(pitch_array),1,p+1)
        plt.plot(np.linspace(0,component-1,component),ci[p],linewidth = 0.5)
        '''

        # 点源随时间仓产生的计数ci曲线
        # fig = plt.figure()
        # plt.subplot(len(pitch_array),1,p+1)
        # s = '(%s,%s)\npitch = %.3f' % (x_pixel1, y_pixel1, pitch_array[p])
        # #fm = '(%s,%s) intenst = %f' % (x_pixel,y_pixel,source_inte)
        # plt.text(-80, 15000, s)
        # #plt.text(720, 50, fm)
        # plot_cim_pixel(x_pixel1, y_pixel1, pixel_range, cim[p], component)
        # 点源随时间仓产生的计数ci曲线 20240427
        # save_path_curve = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\调制模式曲线.pdf'
        # c1_cache = []
        # for i in range(component):
        #     c1_cache.append(cim[p][i][x_pixel1 * pixel_range + y_pixel1])
        # ax = fig_module_curve.add_subplot(len(pitch_array), 1, p+1)
        # ax.set_title(r'$Resolution={:.2f}\, arcsec$'.format(pitch_array[p] / grid_dis * 57.3 * 3600 /2), fontsize=6, fontdict={"verticalalignment":"bottom"}, y=0.7, bbox={'boxstyle':'round', 'facecolor':'white'})
        # if p < len(pitch_array) - 1:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # if p == len(pitch_array)-1:
        #     ax.set_yticks([])
        #     ax.set_xticks(np.linspace(0,component,10), np.int16(module_range/component*np.linspace(0,component,10)), fontsize=6)
        #     ax.set_xlabel(r'$Grid\ angle \, /\, \degree$', fontsize=8)
        # ax.plot(np.linspace(0,component-1,component), c1_cache, linewidth = 0.5, color = 'r')
        # fig_module_curve.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.6)
        # # ax.annotate(r'$Resolution={:.2f}\, arcsec$'.format(pitch_array[p] / grid_dis * 57.3 * 3600/2),
        # #             xy=(0,0), xycoords='axes fraction', xytext=(-0.3,0.5), textcoords='axes fraction', fontsize=8)
        # plt.savefig(save_path_curve, dpi=1000, format='pdf', bbox_inches='tight')

        # # 三维透过率曲线
        # save_path_trans_3d = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\不同空间分辨率双狭缝透过率函数的三维显示.pdf'
        # p1_cache = np.reshape(np.array([p_value[p][0]]), (pixel_range, pixel_range))
        # X,Y = np.mgrid[0:pixel_range, 0:pixel_range]
        # ax_tran_3d = fig_trans_3d.add_subplot(1, len(pitch_array), p+1, projection='3d')
        # ax_tran_3d.set_title(r'${:.2f}\, arcsec$'.format(pitch_array[p] / grid_dis * 57.3 * 3600/2), fontsize=8)
        # ax_tran_3d.set_xticks([])
        # ax_tran_3d.set_yticks([])
        # ax_tran_3d.zaxis.set_tick_params(labelsize=6)
        # ax_tran_3d.set_zlim(0, 1)
        # ax_tran_3d.plot_surface(X, Y, p1_cache, cmap = 'rainbow')
        # ax_tran_3d.view_init(elev=10, azim=-10)
        # fig_trans_3d.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.4)
        # plt.savefig(save_path_trans_3d, dpi=1000, format='pdf', bbox_inches='tight')

        '''
        plt.figure(3)
        plt.subplot(len(pitch_array),1,p+1)
        plot_cim(x_pixel1, y_pixel1, pixel_range, cim[p], component)
        
        plt.figure(4)
        plt.subplot(len(pitch_array),1,p+1)
        plot_cim(x_pixel2, y_pixel2, pixel_range, cim[p], component)
        '''
        
        '''
        #某时间仓下pim条纹，透过率
        plt.figure(4)
        plt.imshow(np.reshape(pim[p][0],(pixel_range,pixel_range)),cmap = 'gray')
        '''
        
        
        #backprojection
        # f1 = backprojection(p_flat[p], ci[p])
        # # f1 = backprojection(p_value[p], ci[p])#多参数狭缝叠加时，不进行归一化处理pim存在问题
        # if farray.any():
        #     farray = np.concatenate((farray, f1[:, :, np.newaxis]), axis=2)
        # else:
        #     farray = deepcopy(f1[:, :, np.newaxis])
        # f = backprojection(p[p], ci[p])
        #f += backprojection(p_value[p], ci[p])
        #f[31][31]=0

        # 单节距双狭缝反投影叠加过程绘图
        save_path = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\单节距反投影累加过程图.jpg'
        fig = plt.figure('back')
        fa = np.zeros_like(p_flat)

        for i in range(component):
            fa[p, i] = ci[p][i] * p_flat[p][i] / time_intev
            axc = fig.add_subplot(20, 10, i * 5 + 1)
            axc.set_xlabel('')
            axc.set_ylabel('')
            axc.set_xticks([])
            axc.set_yticks([])
            plt.setp(axc.spines.values(), linewidth=0.5)
            axc.imshow(p_value[p][i], cmap='gray')
            axc.imshow(init_img, cmap='rainbow', alpha=init_img_alpha)

            axc1 = fig.add_subplot(20, 10, i * 5 + 2)
            axc1.set_xlabel('')
            axc1.set_ylabel('')
            axc1.set_xticks([])
            axc1.set_yticks([])
            plt.setp(axc1.spines.values(), linewidth=0.5)
            axc1.imshow(ci[p][i]/np.max(ci[p][:]) * np.ones((pixel_range,pixel_range)), vmin = 0, vmax = 1, cmap='gray')
            axc2 = fig.add_subplot(20, 10, i * 5 + 3)
            axc2.set_xlabel('')
            axc2.set_ylabel('')
            axc2.set_xticks([])
            axc2.set_yticks([])
            plt.setp(axc2.spines.values(), linewidth=0.5)
            axc2.imshow(fa[p, i, :, :], vmin = 0, vmax = np.max(fa[p, :, :, :]), cmap='gray')
            axc3 = fig.add_subplot(20, 10, i * 5 + 4)
            axc3.set_xlabel('')
            axc3.set_ylabel('')
            axc3.set_xticks([])
            axc3.set_yticks([])
            plt.setp(axc3.spines.values(), linewidth=0.5)
            fa1[p] = fa1[p] + fa[p, i, :, :]
            axc3.imshow(fa1[p, :, :], cmap='gray')

        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=-0.9, hspace=0)
        # plt.savefig(save_path, dpi=1000, format='jpg', bbox_inches='tight')

        # 不同节距狭缝成像效果
        save_path_single_grid = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\不同空间分辨率狭缝的反投影成像图.pdf'
        fig_single = plt.figure('single slit projection')
        fa = np.zeros_like(p_flat)
        for i in range(component):
            fa[p, i] = ci[p][i] * p_flat[p][i] / time_intev
            fa1[p] = fa1[p] + fa[p, i, :, :]
        ax_sing = fig_single.add_subplot(2,np.int8(len(pitch_array)/2),p+1)
        plt.setp(ax_sing.spines.values(), linewidth=0.5)
        ax_sing.set_title(r'${:.2f}\, arcsec$'.format(pitch_array[p] / grid_dis * 57.3 * 3600/2), fontsize=8)
        ax_sing.set_xticks([])
        ax_sing.set_yticks([])
        ax_sing.imshow(fa1[p, :, :], cmap='gray')
        fig_single.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace=-0.5)
        # plt.savefig(save_path_single_grid, dpi=1000, format='pdf', bbox_inches='tight')

    # 多节距狭缝叠加成像效果
    fa_accu = reduce(lambda x,y:x+y, fa1)
    save_path_accu = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\反投影成像叠加图.pdf'
    fig_accu = plt.figure('accumulated slit projection')
    ax_accu = fig_accu.add_subplot(1, 2, 2)
    ax_accu.set_xlabel('(b)', fontsize=10, fontproperties='times new roman', fontweight='bold')
    ax_accu.xaxis.set_tick_params(labelsize=8)
    ax_accu.yaxis.set_tick_params(labelsize=8)
    ax_accu.imshow(fa_accu, cmap='rainbow')
    ax_initimg = fig_accu.add_subplot(1, 2, 1)
    ax_initimg.set_xlabel('(a)', fontsize=10, fontproperties='times new roman', fontweight='bold')
    ax_initimg.xaxis.set_tick_params(labelsize=8)
    ax_initimg.yaxis.set_tick_params(labelsize=8)
    ax_initimg.imshow(init_img, cmap='rainbow')
    # plt.savefig(save_path_accu, dpi=1000, format='pdf', bbox_inches='tight')

    # save_path1 = (r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\初始图像.jpg')
    # fig = plt.figure()
    # ax_init = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # ax_init.set_xlabel('')
    # ax_init.set_ylabel('')
    # ax_init.set_xticks([])
    # ax_init.set_yticks([])
    # ax_init.imshow(init_img, cmap='rainbow')
    # plt.savefig(save_path1, dpi=200, format='jpg', bbox_inches='tight')
    #
    # save_path2 = (r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\单狭缝叠加.jpg')
    # fig = plt.figure()
    # ax_sigleslit = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax_sigleslit.set_xlabel('')
    # ax_sigleslit.set_ylabel('')
    # ax_sigleslit.set_xticks([])
    # ax_sigleslit.set_yticks([])
    # ax_sigleslit.imshow(fa1[p, :, :], cmap='gray')
    # plt.savefig(save_path2, dpi=200, format='jpg', bbox_inches='tight')
    #
    #
    # f = np.sum(farray, axis=2)
    # fig = plt.figure(r'{}'.format(p))
    # for i in range(len(pitch_array)):
    #     axf = fig.add_subplot(3, np.int8(np.ceil(len(pitch_array)/3)), i+1)
    #     axf.imshow(farray[:,:,i], cmap='rainbow')
    #
    # fcache = np.zeros((pixel_range, pixel_range))
    # f2 = np.zeros_like(farray)
    # fig = plt.figure()
    # for i in range(len(pitch_array)):
    #     fcache = farray[:,:,i] + fcache
    #     f2[:,:,i] = fcache
    #     axf = fig.add_subplot(3, np.int8(np.ceil(len(pitch_array) / 3)), i + 1)
    #     axf.imshow(f2[:, :, i], cmap='rainbow')
    #
    # fig = plt.figure()
    # ax5 = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax5.imshow(f2[:, :, -1], cmap = 'rainbow')
    #
    # fig = plt.figure()
    # ax6 = fig.add_axes([0.05,0.05,0.9,0.9], projection='3d')
    # #x = linspace(0,pixel_range,1)
    # #y = linspace(0,pixel_range,1)
    # X,Y = np.mgrid[0:pixel_range:1,0:pixel_range:1]
    # ax6.plot_surface(X, Y, f, cmap='rainbow', linewidth=0, antialiased=False)

    synPSF(pitch_array, np.pi/3600)

    plt.show()
    