#usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 20220901

@author: kakamesyli
'''

from matplotlib import pyplot as plt
import numpy as np
from backprojection import backproject, Clean
from copy import deepcopy
import random
from scipy.interpolate import interp2d
from cmath import pi
from _functools import reduce
from matplotlib.ticker import MultipleLocator


grid_dis = 1.2e3
init_phase_pim = 0
uvplaneRange = 2**7

class DirectReverse(backproject.Pim):
    def __init__(self):
        self.pitchArray = np.array([36e-3,52e-3,76e-3,108e-3,156e-3,224e-3,344e-3,524e-3,800e-3,1224e-3])
        #self.pitchArray = np.array([360e-3])
        #self.angleArray = np.delete(np.linspace(0.5,pi,2),-1)
        self.angleArray = [[20,65,110,155],
                            [0,36,72,108,144],
                            [27,63,99,135,171],
                            [18,54,90,126,162],
                            [9,45,81,117,153],
                            [0,36,72,108,144],
                            [18,54,90,126,162],
                            [0,36,72,108,144],
                            [18,78,138],
                            [48,108,168,168+120,168+240]]
        
        #self.pitchArray = np.array([36e-3])
        #self.angleArray = [np.linspace(0,180*(90-1)/90,360)]
        self.uvpointNum = uvplaneRange
        self.uvplaneRange = uvplaneRange
        
        self.COpen = 0
        
    
    
    def UVPlaneCreate(self):
        Uaxes = np.array([])
        Vaxes = np.array([])
        for i in range(len(self.pitchArray)):
            UaxesCache = np.cos(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]
            VaxesCache = np.sin(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]
            #UaxesCache = np.cos(np.array(self.angleArray)) / self.pitchArray[i]
            #VaxesCache = np.sin(np.array(self.angleArray)) / self.pitchArray[i]
            if Uaxes.any() or Vaxes.any():
                Uaxes = np.concatenate((Uaxes,UaxesCache),axis=0) 
                Vaxes = np.concatenate((Vaxes,VaxesCache),axis=0)
            else:
                Uaxes = UaxesCache
                Vaxes = VaxesCache
        URange = np.max(Uaxes)*2
        VRange = np.max(Vaxes)*2
        UVRange = np.max([URange,VRange])
        delteUV = UVRange/self.uvpointNum
        UPos = np.round(Uaxes / delteUV)
        VPos = np.round(Vaxes / delteUV)
        UVPlane = np.zeros([self.uvplaneRange,self.uvplaneRange])
        uvshift = self.uvpointNum/2-1
        uvshiftextent = (self.uvplaneRange - self.uvpointNum) / 2
        
        '''
        for i in range(len(VPos)):
            for j in range(len(UPos)):
                UVPlane[int(UPos[j]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = 1
        '''
        
        for i in range(len(VPos)):
            UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = 1      
        
        
        return UPos+uvshift+uvshiftextent,VPos+uvshift+uvshiftextent,UVPlane
        
def ImageCreate():
    SourceDistri = backproject.create_init_img(uvplaneRange)
    return SourceDistri      

class IntenseTrans(DirectReverse):
    def __init__(self, img):
        super().__init__()
        self.calCOpen(img)
        #self.calCOpenCouple(img)
        
    
        
    def createUVCountEachGrid(self,imagePolarIntense,pitch,grid_angle)->np.ndarray:
        CUVCount = 0
        for pixel in imagePolarIntense:
            CUVCount = CUVCount + pixel[2] * backproject.calUVTransform(pitch, grid_angle, pixel[0], pixel[1])
            '''
            if CUVCount.any():
                CUVCount = np.concatenate((CUVCount,backproject.calUVTransform(pitch, grid_angle, pixel[0], pixel[1])),axis=1)
            else:
                CUVCount = CUVCount + backproject.calUVTransform(pitch, grid_angle, pixel[0], pixel[1])
        CUVCountAll = np.sum(CUVCount,1)
        '''
        CUVCountFlat = CUVCount - self.COpen
        
        return CUVCountFlat#[2,1]
    
    def createUVCountEachGridCouple(self,imagePolarIntense,pitch,grid_angle)->np.ndarray:
        CUVCount = 0
        for pixel in imagePolarIntense:
            CUVCount = CUVCount + pixel[2] * backproject.calUVTransformCouple(pitch, grid_angle, pixel[0], pixel[1])

        CUVCountFlat = CUVCount - self.COpen
        
        return CUVCountFlat#[4,N1]

    def calImagePolarCoordinate(self, img):
        ImagePolarCoordinateIntense = backproject.calPolarCoordinate(img)
        return ImagePolarCoordinateIntense
    
    def calCOpen(self, img):
        ImagePolarCoordinateIntense = self.calImagePolarCoordinate(img)
        self.COpen = np.sum(ImagePolarCoordinateIntense[:,2])/4
        
    def calCOpenCouple(self, img):
        ImagePolarCoordinateIntense = self.calImagePolarCoordinate(img)
        self.COpen = np.sum(ImagePolarCoordinateIntense[:,2])/4/2
    
    def createUVCount(self,imgPolarIntense,pitchArray,angleArray):
        CountString = np.array([])
        for num in range(len(pitchArray)):
            if CountString.any():
                CountString = np.concatenate((CountString,self.createUVCountEachGrid(imgPolarIntense, pitchArray[num], angleArray[num])),axis=1)
            else:
                CountString = self.createUVCountEachGrid(imgPolarIntense, pitchArray[num], angleArray[num])
        return CountString
    
    def createUVCountCouple(self,imgPolarIntense,pitchArray,angleArray):
        CountString = np.array([])
        for num in range(len(pitchArray)):
            if CountString.any():
                CountString = np.concatenate((CountString,self.createUVCountEachGridCouple(imgPolarIntense, pitchArray[num], angleArray[num])),axis=1)
            else:
                CountString = self.createUVCountEachGridCouple(imgPolarIntense, pitchArray[num], angleArray[num])
        return CountString#[4*N]
        
    def UVPlaneCreate(self,*args):#输入是countstring序列，表达采集到的uv分布
        Uaxes = np.array([])
        Vaxes = np.array([])
                
        for i in range(len(self.pitchArray)):
            UaxesCache = np.cos(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]
            VaxesCache = np.sin(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]
            UaxesCache1 = -np.cos(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]
            VaxesCache1 = -np.sin(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]
            UCache = np.array([UaxesCache,UaxesCache1]).reshape(-1)
            VCache = np.array([VaxesCache,VaxesCache1]).reshape(-1)
            if Uaxes.any() or Vaxes.any():
                Uaxes = np.concatenate((Uaxes,UCache),axis=0) 
                Vaxes = np.concatenate((Vaxes,VCache),axis=0)
            else:
                Uaxes = UCache
                Vaxes = VCache
        
        FSampleFreq = np.max([np.max(Uaxes),np.max(Vaxes)])/self.uvpointNum
        
        scale = ShaFunctionScaleModify()
        
        URange = np.max(Uaxes)*2
        VRange = np.max(Vaxes)*2
        UVRange = np.max([URange,VRange])
        delteUV = UVRange/self.uvpointNum
        UPos = np.round(Uaxes/ delteUV)
        VPos = np.round(Vaxes/ delteUV)
        UPosScale = np.round(Uaxes * scale / delteUV)
        VPosScale = np.round(Vaxes * scale/ delteUV)
        UVPlane = np.zeros([self.uvplaneRange,self.uvplaneRange],dtype=float)
        UVPlaneScale = np.zeros([self.uvplaneRange,self.uvplaneRange])
        UVPlaneReal = np.zeros([self.uvplaneRange,self.uvplaneRange],dtype=complex)
        UVPlaneImag = np.zeros([self.uvplaneRange,self.uvplaneRange],dtype=complex)
        uvshift = self.uvpointNum/2-1
        uvshiftextent = (self.uvplaneRange - self.uvpointNum) / 2
        
        
        
        if args:
            uvValue = args[0]#uvValue数据结构是[N,2],N是采样坐标数
            for i in range(len(VPos)):
                UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = uvValue[0,i] - uvValue[1,i]*1j#减号说明是exp(-2pi)
        else:
            for i in range(len(VPos)):
                if( int(UPosScale[i]+uvshift+uvshiftextent) < self.uvpointNum and int(UPosScale[i]+uvshift+uvshiftextent) > 0 and 
                    int(VPosScale[i]+uvshift+uvshiftextent) < self.uvpointNum and int(VPosScale[i]+uvshift+uvshiftextent) > 0):
                    UVPlaneScale[int(UPosScale[i]+uvshift+uvshiftextent),int(VPosScale[i]+uvshift+uvshiftextent)] = 1
                UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = 1
        
        
        '''
        if args:
            uvValue = args[0]
            for i in range(len(VPos)):
                UVPlaneReal[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = uvValue[0,i]
                UVPlaneImag[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = -uvValue[1,i]
        else:
            for i in range(len(VPos)):
                UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = 1
        '''
        
        #return UPos+uvshift+uvshiftextent,VPos+uvshift+uvshiftextent,UVPlane
        return UPos, VPos, UVPlane, UVPlaneScale
    
    
    def UVPlaneCreate1(self,*args):#采用uv真实值数据
        Uaxes = np.array([])
        Vaxes = np.array([])
                
        for i in range(len(self.pitchArray)):
            UaxesCache = np.cos(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]/(57.3*3)
            VaxesCache = np.sin(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]/(57.3*3)
            UaxesCache1 = -np.cos(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]/(57.3*3)
            VaxesCache1 = -np.sin(np.array(self.angleArray[i])*pi/180) / self.pitchArray[i]/(57.3*3)
            UCache = np.array([UaxesCache,UaxesCache1]).reshape(-1)
            VCache = np.array([VaxesCache,VaxesCache1]).reshape(-1)
            if Uaxes.any() or Vaxes.any():
                Uaxes = np.concatenate((Uaxes,UCache),axis=0) 
                Vaxes = np.concatenate((Vaxes,VCache),axis=0)
            else:
                Uaxes = UCache
                Vaxes = VCache
        
        FSampleFreq = np.max([np.max(Uaxes),np.max(Vaxes)])/self.uvpointNum
        
        scale = ShaFunctionScaleModify()
        
        URange = np.max(Uaxes)*2
        VRange = np.max(Vaxes)*2
        UVRange = np.max([URange,VRange])
        delteUV = UVRange/self.uvpointNum
        #position coordinate
        #UPos = np.round(Uaxes/ delteUV)
        #VPos = np.round(Vaxes/ delteUV)
        #real value
        UPos = Uaxes
        VPos = Vaxes
        UPosScale = np.round(Uaxes * scale / delteUV)
        VPosScale = np.round(Vaxes * scale/ delteUV)
        UVPlane = np.zeros([self.uvplaneRange,self.uvplaneRange],dtype=float)
        UVPlaneScale = np.zeros([self.uvplaneRange,self.uvplaneRange])
        UVPlaneReal = np.zeros([self.uvplaneRange,self.uvplaneRange],dtype=complex)
        UVPlaneImag = np.zeros([self.uvplaneRange,self.uvplaneRange],dtype=complex)
        uvshift = self.uvpointNum/2-1
        uvshiftextent = (self.uvplaneRange - self.uvpointNum) / 2
        
        
        
        if args:
            uvValue = args[0]#uvValue数据结构是[N,2],N是采样坐标数
            for i in range(len(VPos)):
                UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = uvValue[0,i] - uvValue[1,i]*1j#减号说明是exp(-2pi)
        else:
            for i in range(len(VPos)):
                if( int(UPosScale[i]+uvshift+uvshiftextent) < self.uvpointNum and int(UPosScale[i]+uvshift+uvshiftextent) > 0 and 
                    int(VPosScale[i]+uvshift+uvshiftextent) < self.uvpointNum and int(VPosScale[i]+uvshift+uvshiftextent) > 0):
                    UVPlaneScale[int(UPosScale[i]+uvshift+uvshiftextent),int(VPosScale[i]+uvshift+uvshiftextent)] = 1
                UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = 1
        
        
        '''
        if args:
            uvValue = args[0]
            for i in range(len(VPos)):
                UVPlaneReal[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = uvValue[0,i]
                UVPlaneImag[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = -uvValue[1,i]
        else:
            for i in range(len(VPos)):
                UVPlane[int(UPos[i]+uvshift+uvshiftextent),int(VPos[i]+uvshift+uvshiftextent)] = 1
        '''
        
        #return UPos+uvshift+uvshiftextent,VPos+uvshift+uvshiftextent,UVPlane
        return UPos, VPos, UVPlane, UVPlaneScale
    def InverseCountString(self, *args):
        ICountString = np.array([])
        if args:
            countString = args[0]
            for i in range(countString.shape[1]):
                if ICountString.any():
                    ICountString = np.append(ICountString, [countString[0,i] - countString[1,i]*1j])
                else:
                    ICountString = countString[0,i] - countString[1,i]*1j#组成Count是exp(-2pi)
        return ICountString#[N,]
    
    
    def InverseCountStringCouple(self, *args):
        ICountString = np.array([])
        if args:
            countString = args[0]
            for i in range(countString.shape[1]):
                if ICountString.any():
                    ICountString = np.concatenate((ICountString, [[countString[0,i] - countString[1,i]*1j], [countString[2,i] - countString[3,i]*1j]]), axis=1)
                else:
                    ICountString = np.array( [[countString[0,i] - countString[1,i]*1j], [countString[2,i] - countString[3,i]*1j]] )
        return ICountString#[2,N]
            
    def UVInverseSinglePixel(self, ICountString, pixel, pitchArray, angleArray):#通过计数C反演源图I
        pixelIntense = ICountString * (self.UVExpress(pitchArray, angleArray, pixel[0], pixel[1])[0,:]
                                        + self.UVExpress(pitchArray, angleArray, pixel[0], pixel[1])[1,:]*1j)
        
        '''
        for icountstring in ICountString:
            pixelIntense = icountstring * (self.UVExpress(pitchArray, angleArray, pixel[0], pixel[1])[0]
                                           + self.UVExpress(pitchArray, angleArray, pixel[0], pixel[1])[1]*1j)
        '''
        return np.sum(pixelIntense)
    
    def UVInverseSinglePixelCouple(self, ICountString, pixel, pitchArray, angleArray):#通过计数C反演源图I
        pixelIntense = ICountString[0,:] * (self.UVExpressCouple(pitchArray, angleArray, pixel[0], pixel[1])[0] 
                                            + self.UVExpressCouple(pitchArray, angleArray, pixel[0], pixel[1])[1]*1j)
        + ICountString[1,:] * (self.UVExpressCouple(pitchArray, angleArray, pixel[0], pixel[1])[2] 
                               + self.UVExpressCouple(pitchArray, angleArray, pixel[0], pixel[1])[3]*1j)
        
        '''
        for icountstring in ICountString:
            pixelIntense = icountstring * (self.UVExpress(pitchArray, angleArray, pixel[0], pixel[1])[0]
                                           + self.UVExpress(pitchArray, angleArray, pixel[0], pixel[1])[1]*1j)
        '''
        return np.sum(pixelIntense)
    
    def ImageInverse(self, iCountString, imgPolarInput, pitchArray, angleArray):#反演图像
        imgstring = np.array([])
        for pixel in imgPolarInput:
            if imgstring.any():
                imgstring = np.append(imgstring, (self.UVInverseSinglePixel(iCountString, pixel, pitchArray, angleArray)))

            else:
                imgstring = self.UVInverseSinglePixel(iCountString, pixel, pitchArray, angleArray)

        return imgstring.reshape(self.uvpointNum, self.uvpointNum)
    
    def ImageInverseCouple(self, iCountString, imgPolarInput, pitchArray, angleArray):#反演图像
        imgstring = np.array([])
        for pixel in imgPolarInput:
            if imgstring.any():
                imgstring = np.append(imgstring, (self.UVInverseSinglePixelCouple(iCountString, pixel, pitchArray, angleArray)))

            else:
                imgstring = self.UVInverseSinglePixelCouple(iCountString, pixel, pitchArray, angleArray)

        return imgstring.reshape(self.uvpointNum, self.uvpointNum)
    
    def UVExpressSingle(self, pitch, grid_angle, source_theta, source_phai):#计算逆变换中的exp表达
        K = (2*pi/pitch)
        c = grid_dis * np.tan(source_theta) * np.sin(source_phai - grid_angle)
        UExpress = ((2/pi**2)+2/(3*pi)**2)*np.cos(K*c - init_phase_pim)
        VExpress = ((2/pi**2)+2/(3*pi)**2)*np.sin(K*c - init_phase_pim)
        return np.array([UExpress,VExpress])#实际上是uv的exp量表达
    
    def UVExpressSingleCouple(self, pitch, grid_angle, source_theta, source_phai):#计算逆变换中的exp表达
        K = (2*pi/pitch)
        c = grid_dis * np.tan(source_theta) * np.sin(source_phai - grid_angle)
        cCouple = grid_dis * np.tan(source_theta) * np.sin(source_phai - grid_angle -pi)
        UExpress = (2/pi**2)*np.cos(K*c - init_phase_pim)
        VExpress = (2/pi**2)*np.sin(K*c - init_phase_pim)
        UExpressCouple = (2/pi**2)*np.cos(K*cCouple - init_phase_pim)
        VExpressCouple = (2/pi**2)*np.sin(K*c - init_phase_pim)
        return np.array([UExpress, VExpress, UExpressCouple, VExpressCouple])#实际上是uv的exp量表达 
    
    def UVExpress(self, pitchArray, angleArray, source_theta, source_phai):
        uvexpress = np.array([])
        for num in range(len(pitchArray)):
            if uvexpress.any():
                uvexpress = np.concatenate((uvexpress, self.UVExpressSingle(pitchArray[num], angleArray[num], source_theta, source_phai)),axis=1)
            else:
                uvexpress = self.UVExpressSingle(pitchArray[num], angleArray[num], source_theta, source_phai)
        return uvexpress#[N*2]structure
    
    def UVExpressCouple(self, pitchArray, angleArray, source_theta, source_phai):
        uvexpress = np.array([])
        for num in range(len(pitchArray)):
            if uvexpress.any():
                uvexpress = np.concatenate((uvexpress, self.UVExpressSingleCouple(pitchArray[num], angleArray[num], source_theta, source_phai)),axis=1)
            else:
                uvexpress = self.UVExpressSingleCouple(pitchArray[num], angleArray[num], source_theta, source_phai)
        return uvexpress#[4*N]structure
        
class beamxyCreate(DirectReverse):
    def setBeam(self,BeamuvIn:np.ndarray)->np.ndarray:
        Beamuv = deepcopy(BeamuvIn, memo=None, _nil=[])
        return Beamuv
    def beamCal(self,BeamuvIn):
        Beamuv = self.setBeam(BeamuvIn)
        #beamxy = np.fft.ifft2(Beamuv,s=None,axes=(-2,-1),norm=None)
        beamxy = np.fft.fftshift(np.fft.ifft2(Beamuv,s=None,axes=(-2,-1),norm=None))
        return np.abs(beamxy)


def CreaterandomBeamuv(sample_num,beamShape):
    Beam = np.zeros(beamShape**2)
    for i in range(sample_num):
        Beam[i+np.int(np.floor(random.randrange(1,beamShape**2-1)))] = 1
    return Beam.reshape(beamShape,beamShape)

def Interpolation(beam, *args):
    if args:
        interScale = args[0]
    else:
        interScale = 15
    x = np.linspace(0,beam.shape[0]-1,beam.shape[0])
    y = np.linspace(0,beam.shape[1]-1,beam.shape[1])
    xp = np.linspace(0,beam.shape[0]-1,beam.shape[0]*interScale)
    yp = np.linspace(0,beam.shape[1]-1,beam.shape[1]*interScale)
    beamValue = interp2d(x,y,beam,kind='linear')
    beamInterpolation = beamValue(xp,yp)
    return beamInterpolation


def createFFTShaString(string,upos,vpos,posnum):
    ShaString = np.array([])
    for i in range(posnum):
        if ShaString.any():
            ShaString = np.concatenate((ShaString, [string[int(upos[i]),int(vpos[i])]]),axis=0)
        else:
            ShaString = np.array([string[int(upos[i]),int(vpos[i])]])
    return ShaString


def ShaFunctionScaleModify():#修正sha函数与image直接FFT频域图像的比例
    fftrange = 1/15
    uvrange = 1/3.09
    
    return uvrange/fftrange



if __name__ == "__main__":
    
    #create rand beam matrix
    '''
    Beamshape = 512
    BeamuvIn = CreaterandomBeamuv(2, Beamshape)
    '''
    
    '''
    #crearte uv beam
    UVPLANE = DirectReverse()
    upos,vpos,uvplane = UVPLANE.UVPlaneCreate()
    b = beamxyCreate()
    beamxy = b.beamCal(uvplane)
    beamxyInterpolation = Interpolation(beamxy)
    
    fig = plt.figure(1)
    ax1 = fig.add_axes([0,0,1,1])
    ax1.set_xlim(0,beamxy.shape[0])
    ax1.set_ylim(0,beamxy.shape[1])
    ax1.scatter(vpos,upos,color='black',marker='x')
    #ax1.imshow(uvplane,cmap='binary',marker='x')
    #ax1.hlines(y=beamxy.shape[0]/2,xmin=0,xmax=beamxy.shape[0],linestyle='dotted')
    #ax1.vlines(x=beamxy.shape[0]/2,ymin=0,ymax=beamxy.shape[0],linestyle='dotted')
    fig = plt.figure(2)
    ax2 = fig.add_axes([0,0,1,1])
    ax2.imshow(beamxy,cmap='gray')
    fig = plt.figure(3)
    ax3 = fig.add_axes([0,0,1,1])
    ax3.set_xlim(beamxyInterpolation.shape[0]/2-400,beamxyInterpolation.shape[0]/2+400)
    ax3.set_ylim(beamxyInterpolation.shape[0]/2-400,beamxyInterpolation.shape[0]/2+400)
    ax3.imshow(beamxyInterpolation,cmap='gray')
    '''
    
    
    ImageTest = ImageCreate()
    ImageTestPolar = backproject.calPolarCoordinate(ImageTest)
    UVPLANE = DirectReverse()
    intensetrans = IntenseTrans(ImageTest)
    #CountString = intensetrans.createUVCount(ImageTestPolar, UVPLANE.pitchArray, UVPLANE.angleArray)
    #upos,vpos,beamuvplane,beamuvplanescale = intensetrans.UVPlaneCreate()
    upos,vpos,beamuvplane,beamuvplanescale = intensetrans.UVPlaneCreate1()
    dirtyBeam = np.fft.fftshift(np.fft.ifft2(beamuvplane))
    #upos,vpos,uvplane = intensetrans.UVPlaneCreate(CountString)
    #ImageInverseUV = np.fft.fftshift(np.fft.ifft2(uvplane))
    
    
    #CountString->ICountString->UVExpressSingle->UVExpress->UVsinglePixel->ImageInverse
    
    '''
    #coupled
    CountString = intensetrans.createUVCountCouple(ImageTestPolar, UVPLANE.pitchArray, UVPLANE.angleArray)
    iCountString = intensetrans.InverseCountStringCouple(CountString)
    dirtyMap = intensetrans.ImageInverseCouple(iCountString, ImageTestPolar, UVPLANE.pitchArray, UVPLANE.angleArray)
    '''
    
    '''
    CountString = intensetrans.createUVCount(ImageTestPolar, UVPLANE.pitchArray, UVPLANE.angleArray)
    iCountString = intensetrans.InverseCountString(CountString)
    dirtyMap = intensetrans.ImageInverse(iCountString, ImageTestPolar, UVPLANE.pitchArray, UVPLANE.angleArray)
    '''
    

    
    '''
    b = beamxyCreate()
    dirtyMap = b.beamCal(uvplane)
    dirtyMapReal = b.beamCal(uvplanereal)
    dirtyMapImag = b.beamCal(uvplaneimag)
    dirtyMap = (dirtyMapReal+dirtyMapImag)/2
    '''
    
    '''
    ImageFFT = np.fft.fft2(ImageTest)
    ImageFFTShift = np.fft.fftshift(ImageFFT)
    ImageInverse = np.fft.fftshift(np.fft.ifft2(ImageFFT))
    upos,vpos,Shafunction,ShafunctionScale = intensetrans.UVPlaneCreate()
    DirtyFFTSha = ImageFFTShift * Shafunction.T
    ImageInverseSha = np.fft.ifft2(DirtyFFTSha)
    '''
    
    #ShaString = createFFTShaString(ImageFFTSha,upos,vpos,len(vpos))
    #UVString = CountString[0,:] + CountString[1,:]*1j
    
    
    
    '''
    #clean
    threShould = np.max(np.abs(dirtyMap))/1.8
    iteralLimit = 1e2
    cleanOp = Clean.Clean(dirtyMap, dirtyBeam, threShould)
    cleanMap = cleanOp.cleanAlgorithm()
    fig = plt.figure('CleanMap')
    ax5 = fig.add_axes([0.05,0.05,0.9,0.9])
    ax5.imshow(np.abs(cleanMap),cmap='gray')
    CleanBeam = cleanOp.cleanBeam(uvplaneRange,0.5)
    CleanBeamFFT = np.fft.fftshift(np.fft.fft2(CleanBeam))
    CleanFFTSha = ImageFFTShift * CleanBeamFFT
    fig = plt.figure('cleanbeam')
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ax.imshow(CleanBeam, cmap='gray')
    fig = plt.figure('cleanbeamFFT')
    ax = fig.add_axes([0.2,0.1,0.6,0.8])
    ax.set_xlabel('U mm-1')
    ax.set_ylabel('V mm-1')
    ax.imshow(np.abs(CleanBeamFFT), cmap='cividis')
    fig = plt.figure('cleanFFTsha')
    ax = fig.add_axes([0.2,0.1,0.6,0.8])
    ax.set_xlabel('U mm-1')
    ax.set_ylabel('V mm-1')
    ax.imshow(np.abs(CleanFFTSha), cmap='cividis')
    '''
    
    
    '''
    fig = plt.figure('dirtybeam')
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ax.imshow(np.abs(dirtyBeam),cmap='gray')
    
    
    fig = plt.figure('dirtyFFTsha')
    ax = fig.add_axes([0.2,0.1,0.6,0.8])
    ax.set_xlabel('U mm-1')
    ax.set_ylabel('V mm-1')
    im = ax.imshow(np.abs(DirtyFFTSha),cmap='cividis')
    #cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cax)
    
    
    
    fig = plt.figure('imagefft')
    ax4 = fig.add_axes([0.2,0.1,0.6,0.8])
    ax4.set_xlabel('U')
    ax4.set_ylabel('V')
    ax4.imshow(np.abs(ImageFFTShift), cmap='cividis')
    
    
    fig = plt.figure('shafunction')
    ax5 = fig.add_axes([0.2,0.1,0.6,0.8])
    ax5.set_xlabel('U mm-1')
    ax5.set_ylabel('V mm-1')
    ax5.scatter(vpos, upos, color='black', marker='x')
    
    fig = plt.figure('shafunctionscale')
    ax5 = fig.add_axes([0.2,0.1,0.6,0.8])
    ax5.set_xlabel('U mm-1')
    ax5.set_ylabel('V mm-1')
    ax5.imshow(ShafunctionScale, cmap='cividis')
    
    fig = plt.figure('shafunction-col')
    ax5 = fig.add_axes([0.2,0.1,0.6,0.8])
    ax5.set_xlabel('U mm-1')
    ax5.set_ylabel('V mm-1')
    ax5.imshow(Shafunction.T, cmap='cividis')
    '''
    
    save_path = r'E:\g\DOCTOR\预答辩\各学科群分会学位论文撰写具体要求（2023年8月更新）\模板\Img\Chap_1\HXI光栅的空间频域采样配置点.pdf'
    fig = plt.figure('uvplane')
    ax6 = fig.add_axes([0.05,0.05,0.9,0.9])
    #ax6.imshow(np.abs(uvplane),cmap='gray')
    ax6.set_xlabel('$U\quad arcsec^{-1}$', fontsize=16)
    ax6.set_ylabel('$V\quad arcsec^{-1}$', fontsize=16)
    ax6.xaxis.set_label_coords(0.5,-0.1)
    ax6.xaxis.set_tick_params(labelsize=16)
    ax6.yaxis.set_tick_params(labelsize=16)
    ax6.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax6.yaxis.set_minor_locator(MultipleLocator(0.01))
    
    ax6.scatter(vpos,upos, edgecolors='black',color='w', s=16, marker='o')
    plt.savefig(save_path, dpi=1000, format='pdf', bbox_inches='tight')
    
    '''
    fig = plt.figure('imagesha')
    ax7 = fig.add_axes([0.05,0.05,0.9,0.9])
    ax7.imshow(np.abs(ImageInverseSha),cmap='gray')
    '''
    '''
    fig = plt.figure('imageuv')
    ax7 = fig.add_axes([0,0,1,1])
    ax7.imshow(np.abs(ImageInverseUV),cmap='gray')
    plt.show()
    '''
    
    '''
    fig = plt.figure('InitImage')
    ax4 = fig.add_axes([0.05,0.05,0.9,0.9])
    ax4.imshow(ImageTest,cmap='gray')
    '''
    
    '''
    fig = plt.figure('DirtyMap3D')
    [V,U] = np.mgrid[0:dirtyMap.shape[0]:1,0:dirtyMap.shape[0]:1]
    ax1 = fig.add_axes([0,0,1,1],projection='3d')
    #ax1.set_xlim(0,dirtyMap.shape[0])
    #ax1.set_ylim(0,dirtyMap.shape[1])
    ax1.plot_surface(V,U,np.abs(dirtyMap))
    #ax1.imshow(np.abs(beamuvplane))
    #ax1.hlines(y=dirtyMap.shape[0]/2,xmin=0,xmax=dirtyMap.shape[0],linestyle='dotted')
    #ax1.vlines(x=dirtyMap.shape[0]/2,ymin=0,ymax=dirtyMap.shape[0],linestyle='dotted')
    '''
    
    '''
    fig = plt.figure('Dirtymap')
    ax2 = fig.add_axes([0.05,0.05,0.9,0.9])
    ax2.imshow(np.abs(dirtyMap),cmap='gray')
    '''
    
    
    '''
    fig = plt.figure(3)
    ax3 = fig.add_axes([0,0,1,1])
    ax3.set_xlim(dirtyMap.shape[0]/2-400,dirtyMap.shape[0]/2+400)
    ax3.set_ylim(dirtyMap.shape[0]/2-400,dirtyMap.shape[0]/2+400)
    ax3.imshow(dirtyMap,cmap='gray')
    '''
    
    
    
    plt.show()
    
    