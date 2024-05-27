# usr/bin/env python
# -*- coding:utf-8 -*-
import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import signal
import copy
import cv2

class coded_aperture(object):
    def __init__(self):
        self.rowi = 43
        self.colj = 41
        self.Ci = np.zeros(self.rowi)
        self.Cj = np.zeros(self.colj)

    # 进行Cr和Cs计算
    def set_C(self):
        self.Cj = self.cal_C(self.colj)
        self.Ci = self.cal_C(self.rowi)
    def cal_C(self,rs):
        C = np.zeros(rs)
        x_array = np.linspace(1,rs-1,rs-1)
        x_array1 = np.mod(x_array ** 2, rs)
        for i in range(rs):
            if i in x_array1:
                C[i] = 1
            else:
                C[i] = -1
        return C
    def make_randmask_1d(self,pusnum):
        ind = []
        mask_array = np.zeros(self.colj)
        for i in range(pusnum):
            ind.append(np.random.randint(0,self.colj-1))
        for i in ind:
            mask_array[i] = 1
        return mask_array

    def make_mask(self):
        mask_array = np.zeros([self.rowi, self.colj])
        self.set_C()
        for i in range(0, self.rowi):
            for j in range(0, self.colj):
                if j == 0 and i != 0:
                    mask_array[i,j] = 1
                elif self.Ci[i]*self.Cj[j] == 1:
                    mask_array[i,j] = 1
        return mask_array
    def make_cycmask(self,m,r):
        mstack = m
        mtohalf = m
        if r==1:
            m2 = m
        elif r==2:
            m1 = self.stackhalfmask(mstack, mtohalf, 0)
            m2 = self.stackhalfmask(m1, m1, 1)
        else:
            m1_1 = self.stackmask(m, r-1, 0)
            m1 = self.stackhalfmask(m1_1, mtohalf, 0)
            m2_2 = self.stackmask(m1, r-1, 1)
            m2 = self.stackhalfmask(m2_2, m1, 1)
        return m2
    def stackmask(self,m,r,axs):
        stackm = copy.deepcopy(m)
        for i in range(0, r-1):
            m = np.concatenate((m, stackm), axis=axs)
        return m
    def stackhalfmask(self,mstack,mtohalf,axs):
        if axs==0:
            h = np.int8((mtohalf.shape[0]-1)/2)
            mhalf1 = mtohalf[0:h, :]
            mhalf2 = mtohalf[h:, :]
            m1 = np.concatenate((mhalf2, mstack), axs)
            m2 = np.concatenate((m1, mhalf1), axs)
        elif axs==1:
            h = np.int8((mtohalf.shape[1]-1)/2)
            mhalf1 = mtohalf[:, 0:h]
            mhalf2 = mtohalf[:, h:]
            m1 = np.concatenate((mhalf2, mstack), axs)
            m2 = np.concatenate((m1, mhalf1), axs)
        return m2

    def make_decodemask(self,mask_array):
        decode_mask = mask_array*2-1
        return decode_mask

    def cal_conv(self,m1,m2):
        cov = signal.convolve(m1,m2,mode='full')
        return cov
    def cal_conv2d(self,m1,m2):
        cov2d = signal.convolve2d(m1,m2,mode='full')
        return cov2d

    def sel_correlation_matrix(self,m):
        sel_corr_matrix = signal.convolve2d(m,m,mode='full')
        return sel_corr_matrix
    def cal_fft(self,m):
        fft = np.fft.fft(m)
        return fft
    def cal_fft2(self,m):
        fft2 = fft.fft2(m)
        return fft2
    def addpinholeshape(self,m,**kwargs):
        row,col = m.shape
        mat = np.array([])
        m_col = np.array([])
        r = kwargs['range']
        d = r*2+1
        if kwargs['shape'] == 'square':
            pinhole = np.ones((d,d))
            pinhole_zero = np.zeros((d,d))
        elif kwargs['shape'] == 'circle':
            pinhole = self.make_circle_shape(r)
            pinhole_zero = np.ones((d, d))
        else:
            print('Wrong')
        for i in range(row):
            for j in range(col):
                if m[i,j] == 1 and j==0:
                    m_col = pinhole
                if m[i,j] == 0 and j==0:
                    m_col = pinhole_zero
                elif m[i,j] == 1 and j!=0:
                    m_col = np.concatenate((m_col,pinhole),axis=1)
                elif m[i,j] == 0 and j!=0:
                    m_col = np.concatenate((m_col, pinhole_zero), axis=1)
            if i ==0:
                mat = m_col
            elif i!=0:
                mat = np.concatenate((mat,m_col),axis=0)
        return mat

    def make_circle_shape(self,r):
        d = r*2+1
        m = np.zeros((d,d))
        cen = r
        for i in range(d):
            for j in range(d):
                if np.sqrt((i-cen)**2+(j-cen)**2) <= r:
                    m[i,j] = 1
        return m
    def pad_zeors_1d(self,m,n):
        a = len(m)
        if n > a:
            supplymat = np.zeros(n-a)
            mat = np.concatenate((m,supplymat),axis=0)
        else:
            print('no need to pad')
            mat = m
        return mat

    def cycle_cov1d(self, mat1, mat2, padnum, cycnum):
        mat1pad = np.array([])
        mat2pad = np.array([])
        if padnum >= len(mat1) and padnum >= len(mat2):
            mat1pad = self.pad_zeors_1d(mat1, padnum)
            mat2pad = self.pad_zeors_1d(mat2, padnum)
        else:
            print('pad error')
        # mat1padcyc = self.stackmask(mat1pad, cycnum, 0)
        mat2padcyc = self.stackmask(mat2pad, cycnum, 0)
        cyccov = signal.convolve(mat1pad, mat2padcyc, mode='full')
        cyccov_mainvalue = cyccov[len(mat1pad):2*len(mat1pad)]
        cyccov_maincyc = self.stackmask(cyccov_mainvalue, cycnum, 0)
        return cyccov

    def cal_code(self, mat1, mat2):
        mat = np.zeros([2*self.rowi, 2*self.colj])
        for k in range(mat.shape[0]):
            for l in range(mat.shape[1]):
                midmat = mat1 * mat2[k:k+mat1.shape[0], l:l+mat1.shape[1]]
                mat[k, l] = np.sum(np.sum(midmat))
        return mat

    def cal_decode(self, mat1, mat2):
        mat = np.zeros([self.rowi, self.colj])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                midmat = mat1 * mat2[i:i+mat1.shape[0], j:j+mat1.shape[1]]
                mat[i, j] = np.sum(np.sum(midmat))
        return mat

    def cal_decodeshift(self, mat1, mat2):
        mat = np.zeros([self.rowi, self.colj])
        r1 = mat.shape[0]
        s1 = mat.shape[1]
        r2 = np.uint((mat.shape[0]+1)/2)
        s2 = np.uint((mat.shape[1]+1)/2)
        for i in range(r1):
            for j in range(s2,s1):
                midmat = mat1 * mat2[i:i+mat1.shape[0], j:j+mat1.shape[1]]
                mat[i, j] = np.sum(np.sum(midmat))
        return mat

    def pub_onezerosmat(self, m1):
        mz = np.zeros((2*m1.shape[0], 2*m1.shape[1]))
        mz[self.rowi:self.rowi+m1.shape[0], self.colj:self.colj+m1.shape[1]] = m1
        return mz

    def img_shift(self, img):
        ax0_cen = np.uint8((img.shape[0]-1)/2)
        ax1_cen = np.uint8((img.shape[1]-1)/2)
        imgup = img[:ax0_cen, :]
        imgdown = img[ax0_cen:, :]
        imgupdown = np.concatenate((imgdown, imgup), axis=0)
        imgleft = imgupdown[:, :ax1_cen]
        imgright = imgupdown[:, ax1_cen:]
        imgleftright = np.concatenate((imgright, imgleft), axis=1)
        return imgleftright

    def relation_rho_phs(self):
        save_path = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URA编码孔径的最佳透光密度曲线.pdf'
        save_path1 = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URA编码孔径与最优孔径密度SRN的对比曲线.pdf'
        psi = np.linspace(0,1,1000)
        a = np.zeros_like(psi)
        b = np.zeros_like(psi)
        r = np.zeros_like(psi)
        s = np.zeros_like(psi)
        sura = np.zeros_like(psi)
        rt = np.zeros_like(psi)
        rho = np.array([])
        ratio = np.array([])
        xi = np.array([0.001, 0.01, 0.1, 1, 10])
        for e in xi:
            b[:] = psi + e
            a[:] = 1 - psi
            r[:] = (np.sqrt(b**2+a*b) - b) / a
            s[:] = (np.sqrt(1-r) * r) / np.sqrt(a*r**2+b*r)
            sura[:] = (np.sqrt(1 - 0.5) * 0.5) / np.sqrt(a * 0.5 ** 2 + b * 0.5)
            rt[:] = sura/s
            if rho.any():
                rho = np.concatenate((rho, r[:, np.newaxis]), axis=1)
                ratio = np.concatenate((ratio, rt[:, np.newaxis]), axis=1)
            else:
                rho = copy.deepcopy(r[:, np.newaxis])
                ratio = copy.deepcopy(rt[:, np.newaxis])

        fig = plt.figure(r'$\rho$ with $\psi$')
        ax = fig.add_subplot([0.05, 0.05, 0.9, 0.9])
        ax.set_xlabel(r'$\psi$', fontsize = 18)
        ax.set_ylabel(r'$\rho$', fontsize = 18)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        c = ['#FF0000', '#33CC33', '#3399FF', '#000000', '#CC00CC']
        xyd = [[psi[3], rho[3,0]],
               [psi[10], rho[10,1]],
               [psi[15], rho[15,2]],
               [psi[10], rho[10,3]],
               [psi[10], rho[10,4]]]
        xyt = [[psi[3]+0.35, rho[3,0]+0.01],
               [psi[10]+0.35, rho[10,1]+0.035],
               [psi[15]+0.3, rho[15,2]-0.01],
               [psi[10]+0.15, rho[10,3]-0.05],
               [psi[10]+0.17, rho[10,4]-0.03]]
        arrowst = ['arc3',
                   'angle, angleA=0, angleB=135',
                   'angle, angleA=0, angleB=135',
                   'arc3',
                   'arc3']
        for i in range(len(xi)):
            ax.plot(psi, rho[:,i], '-', color = c[i], label = r'$\xi={:.3f}$'.format(xi[i]))
            ax.annotate(r'$\xi \, =\, {:}$'.format(xi[i]), xy=xyd[i], xycoords='data',
                        xytext=xyt[i], textcoords='data',
                        arrowprops=dict(arrowstyle='->', facecolor='black', connectionstyle=arrowst[i]),
                        horizontalalignment='right', verticalalignment='top', fontsize=14)
        # plt.legend(loc='lower right', ncol=1, bbox_to_anchor=(1, 0), fontsize=8, columnspacing=1, labelspacing=0.5,
        #            handlelength=1, handletextpad=0.5, handleheight=0.5, fancybox=True, edgecolor='k')
        plt.savefig(save_path, dpi=1000, format='pdf', bbox_inches='tight')


        fig = plt.figure(r'$snrura$ with $snr$')
        ax1 = fig.add_subplot([0.05, 0.05, 0.9, 0.9])
        ax1.set_xlabel(r'$\psi$', fontsize=18)
        ax1.set_ylabel(r'$Ratio$', fontsize=18)
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        xyd = [[psi[3], ratio[3, 0]],
               [psi[5], ratio[5, 1]],
               [psi[10], ratio[10, 2]],
               [psi[10], ratio[10, 3]],
               [psi[50], ratio[50, 4]]]
        xyt = [[psi[3] + 0.35, ratio[3, 0] + 0.01],
               [psi[5] + 0.35, ratio[5, 1] + 0.02],
               [psi[10] + 0.35, ratio[10, 2] - 0.025],
               [psi[10] + 0.1, ratio[10, 3] - 0.025],
               [psi[50] + 0.33, ratio[50, 4] - 0.08]]
        arrowst = ['arc3',
                   'angle, angleA=0, angleB=135',
                   'angle, angleA=0, angleB=135',
                   'arc3',
                   'arc3']
        for i in range(len(xi)):
            ax1.plot(psi, ratio[:, i], '-', color=c[i], label=r'$\xi={:.3f}$'.format(xi[i]))
            ax1.annotate(r'$\xi \, =\, {:}$'.format(xi[i]), xy=xyd[i], xycoords='data',
                        xytext=xyt[i], textcoords='data',
                        arrowprops=dict(arrowstyle='->', facecolor='black', connectionstyle=arrowst[i]),
                        horizontalalignment='right', verticalalignment='top', fontsize=14)
        plt.savefig(save_path1, dpi=1000, format='pdf', bbox_inches='tight')


def img_read(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return img

def cov2singtest(a,b,k,l):
    s = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            s = s + a[i, j] * b[k-i, l-j]
    return s

if __name__ == '__main__':
    code = coded_aperture()
    mask1 = code.make_mask()
    decodemask1 = code.make_decodemask(mask1)
    # mask1d = code.make_randmask_1d(np.int8(code.r))
    # mask1d = np.ones(5)
    # mask1d2 = np.array([1, 2])
    # mask1dstack = code.stackmask(mask1d,1,0)
    # mask1dcov = code.cal_conv(mask1dstack,mask1dstack)
    # cycmask1d = code.cycle_cov1d(mask1d, mask1d, len(mask1d), 5)
    # mask1fft = code.cal_fft(cycmask1d)

    # mask_pinhole = code.addpinholeshape(mask1, shape='circle', range = 6)
    # mask1 = code.make_cycmask(mask1, 2)
    mask1cyc = code.make_cycmask(mask1, 4)
    decodemask1cyc = code.make_decodemask(mask1cyc)
    mask1cyczero = code.pub_onezerosmat(mask1cyc)
    decodemask1zero = code.pub_onezerosmat(decodemask1)
    # m1 = mask1cyc[0, :]
    # m2 = decodemask1[0, :]
    # c = code.cycle_cov1d(m1, m2, len(m1), 10)
    # psfmat = code.cal_conv2d(mask1, decodemask1)
    # correlationmat = code.sel_correlation_matrix(mask1cyc)
    # mask1fft = code.cal_fft(psfmat)

    # save_path = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URA编码孔径2.pdf'
    # fig = plt.figure()
    # ax1 = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax1.imshow(mask1cyc, cmap='gray')
    # ax2 = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax2.imshow(mask1cyc, cmap='gray')
    # fig = plt.figure('3d')
    # psf = psfmat
    # matplot = np.abs(mask1fft)
    # x = np.linspace(0, matplot.shape[0], matplot.shape[0])
    # y = np.linspace(0, matplot.shape[1], matplot.shape[1])
    # X, Y = np.mgrid[0:matplot.shape[0]:1, 0:matplot.shape[1]:1]
    # ax3 = fig.add_axes([0.05, 0.05, 0.4, 0.9],projection='3d')
    # ax3.plot_surface(X, Y, matplot, cmap='coolwarm', antialiased=False)
    # ax3.bar3d(X.ravel(), Y.ravel(), 0, 0.5, 0.5, matplot.ravel())
    # x0 = np.int8((psf.shape[0]-1)/2)
    # y0 = np.int8((psf.shape[1]-1)/2)
    # ax3.plot(x, matplot[:,y0], zs=-1, zdir='y',color='r')
    # ax3.plot(y, matplot[x0,:], zs=-1, zdir='x',color='g')
    # ax3.bar(x, matplot[:, y0], zs=-1, zdir='y', color='r', width=0.5)
    # ax3.bar(y, matplot[x0, :], zs=-1, zdir='x', color='g', width=0.5)
    # ax4 = fig.add_axes([0.05, 0.05, 0.4, 0.9])
    # ax4.imshow(psf, cmap='coolwarm')
    # ax5 = fig.add_axes([0.5, 0.05, 0.4, 0.9], projection='3d')
    # X1, Y1 = np.mgrid[0:psf.shape[0]:1, 0:psf.shape[1]:1]
    # ax5.plot_surface(X1, Y1, psf, cmap='coolwarm')
    # x1 = np.linspace(0, psf.shape[0], psf.shape[0])
    # y1 = np.linspace(0, psf.shape[1], psf.shape[1])
    # ax5.bar(x1, psf[:, y0], zs=-1, zdir='y', color='r', width=0.5)
    # ax5.bar(y1, psf[x0, :], zs=-1, zdir='x', color='g', width=0.5)
    # ax5.bar3d(X1.ravel(), Y1.ravel(), 0, 0.1, 0.1, psf.ravel())
    # plt.savefig(save_path, dpi=1000, format='pdf', bbox_inches='tight')

    # save_path = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\1d_mask_rand.pdf'
    # fig = plt.figure('1d')
    # ax4 = fig.add_subplot(2,1,1)
    # ax4.bar(range(len(c)),c)
    # ax5 = fig.add_subplot(2,1,2)
    # ax5.bar(range(len(mask1fft)),np.abs(mask1fft))
    # plt.savefig(save_path, dpi = 1000, format='pdf', bbox_inches='tight')

    ag = img_read(r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\编码孔径原始图像混叠.jpg')
    save_path1 = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URA编码孔径原始图像混叠.pdf'
    save_path2 = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URA编码孔径探测器光子累加强度混叠.pdf'
    save_path3 = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URA编码孔径反演图像.pdf'
    save_path4 = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_2\URAPSF.pdf'
    ag_resam = cv2.resize(ag, (2*mask1.shape[1], 2*mask1.shape[0]), interpolation=cv2.INTER_CUBIC)
    Pimg = code.cal_code(ag_resam, mask1cyc)
    Oimg = code.cal_decode(Pimg, decodemask1cyc)
    Oimgshift = code.img_shift(Oimg)
    URAPSF = code.cal_decode(mask1, decodemask1cyc)
    # fig = plt.figure(3)
    # ax6 = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax6.imshow(ag, cmap='gray')
    # plt.savefig(save_path1, dpi=1000, format='pdf', bbox_inches='tight')
    # fig = plt.figure(7)
    # ax7 = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax7.imshow(Pimg, cmap='gray')
    # plt.savefig(save_path2, dpi=1000, format='pdf', bbox_inches='tight')
    # fig = plt.figure(8)
    # ax8 = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax8.imshow(Oimgshift, cmap='gray')
    # plt.savefig(save_path3, dpi=1000, format='pdf', bbox_inches='tight')
    # fig = plt.figure('URApsf')
    # ax6 = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='3d')
    # x = np.linspace(0, URAPSF.shape[0], URAPSF.shape[0])
    # y = np.linspace(0, URAPSF.shape[1], URAPSF.shape[1])
    # X, Y = np.mgrid[0:URAPSF.shape[0]:1, 0:URAPSF.shape[1]:1]
    # ax6.plot_surface(X, Y, URAPSF, cmap='coolwarm', antialiased=False)
    # ax6.set_xlabel('$X$', fontsize=12)
    # ax6.set_ylabel('$Y$', fontsize=12)
    # ax6.set_yticks(np.arange(0,45,1)[0:45:10], np.arange(0,45,1)[0:45:10])
    # ax6.imshow(URAPSF, cmap='gray')
    # plt.savefig(save_path4, dpi=1000, format='pdf', bbox_inches='tight')

    code.relation_rho_phs()

    plt.show()