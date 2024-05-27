#usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2022年9月28日

@author: kakamesyli
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg
from scipy import interpolate
from backprojection import backproject
from scipy.ndimage import interpolation

class Clean(object):
    '''
    classdocs
    '''


    def __init__(self, dirtyMap, dirtyBeam, threShould):
        '''
        Constructor
        '''
        self.dirtymap = dirtyMap
        self.dirtybeam = np.abs(dirtyBeam)
        self.threshould = threShould
        self.size = self.dirtymap.shape[0]
    
    def cleanAlgorithm(self, *Limit):
        loopgain = 2
        residuleImage = np.abs(self.dirtymap)
        cleanMap = np.zeros([residuleImage.shape[0], residuleImage.shape[0]])
        storeiterimg = np.zeros([residuleImage.shape[0], residuleImage.shape[0]])
        iter = 0
        if Limit:
            iteralLimit = Limit[0]
        
        plt.ion()
        fig = plt.figure('clean iter')
        ax1 = fig.add_subplot(1,3,1)
        ax1.set_title('residual image')
        ax2 = fig.add_subplot(1,3,2)
        ax2.set_title('dirtybeam image')
        ax3 = fig.add_subplot(1,3,3)
        ax3.set_title('store image')
        
        '''
        fig = plt.figure('store iter img')
        ax4 = fig.add_axes([0.05,0.05,0.9,0.9])
        '''
        
        '''
        fig = plt.figure('dirtybeam image')
        axresidual = fig.add_axes([0.05,0.05,0.9,0.9])
        '''
        
        
        totalMax =  self.getMax(residuleImage)['maxvalue']
        while ( np.max(residuleImage) >= self.threshould ):
            #plt.clf()
            
            maxPixel =  self.getMax(residuleImage)
            dirtybeamImage = self.calDirtyBeamImage(self.dirtybeam, maxPixel, loopgain)
            residuleImage = residuleImage - dirtybeamImage
            storeImage, cleanImage = self.cleanImage(maxPixel, loopgain)
            cleanMap = cleanMap + cleanImage
            storeiterimg = storeiterimg + storeImage
            #updatePlot(residuleImage, dirtybeamImage, storeImage, storeiterimg, fig, ax1, ax2, ax3, ax4, totalMax)
            #plt.pause(0.0001)
            #axresidual.imshow(residuleImage,cmap='gray',vmin=0,vmax=totalMax)
            
            
            iter += 1
            '''
            if (iter % 10 == 0):
                print(iter)
            '''
            
            
            '''
            if (iter % (iteralLimit/10) == 0):
                print('%d%% has completed...' % (iter / (iteralLimit/100) ))
            elif(iter >= iteralLimit):
                break
            '''
        print(iter)
        plt.ioff()
        
        '''
        fig = plt.figure('residual_after')
        axresidualafter = fig.add_axes([0,0,1,1])
        axresidualafter.imshow(residuleImage)
        '''
        return cleanMap + residuleImage
            
    def getMax(self, image):
        maxValue = np.max(image)
        maxPosition = np.unravel_index(image.argmax(), image.shape)
        return {'maxvalue':maxValue, 'maxposition':maxPosition}
    
    def calDirtyBeamImage(self, dirtybeam, maxpixel, loopgain):
        #dirtybeamImg = np.zeros([dirtybeam.shape[0], dirtybeam.shape[0]])
        #dirtybeamImg[maxpixel['maxposition']] = loopgain * maxpixel['maxvalue']
        dirtybeamImg = loopgain * maxpixel['maxvalue'] * dirtybeam
        shiftpos = np.array((maxpixel['maxposition'])) - np.array([(dirtybeam.shape[0])/2,(dirtybeam.shape[0])/2])
        dirtybeamImg = interpolation.shift(dirtybeamImg, shiftpos, cval=0)
        
        #return np.abs( sg.convolve2d(dirtybeamImg, dirtybeam, mode='same', boundary='fill', fillvalue=0) )
        return dirtybeamImg
    
    def cleanImage(self, maxpixel, loopgain):
        storeimage = np.zeros([self.dirtymap.shape[0], self.dirtymap.shape[0]])
        storeimage[maxpixel['maxposition']] = loopgain * maxpixel['maxvalue']
        cleanrange = 20
        sigma = 2
        cleanbeam = self.cleanBeam(cleanrange, sigma)
        cleanImg = sg.convolve2d(storeimage, cleanbeam, mode='same', boundary='fill', fillvalue=0)
        return storeimage, cleanImg
    
    def cleanBeam(self, range, sig):
        cleanbeam = backproject.create_guass(range, sig)

        return cleanbeam
    
    
        
def updatePlot(residuleImage, dirtybeamImage, storeImage, storeiterimg, fig, ax1, ax2, ax3, ax4, totalmax):
    #fig.clf()
    ax1.imshow(residuleImage, cmap='gray', vmin=0, vmax=totalmax)
    ax2.imshow(dirtybeamImage, cmap='gray')
    ax3.imshow(storeImage, cmap='gray')
    ax4.imshow(storeiterimg, cmap='gray')
    #fig.canvas.draw_idle()
        
        
      