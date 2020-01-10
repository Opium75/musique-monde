
import scipy.io.wavfile as wv
from numpy import fft, empty, array_split, array, abs, append
import numpy as np
#import matplotlib.pyplot as plt
import math as m
import scipy.interpolate as ip



class FftSinusoid:
    # _data
    # _freq
    # since we'll fft.
    def __init__(self, freq, amp, rate, nbSamples):
        omega = 2*m.pi*freq
        x = np.arange(2*nbSamples-1)
        sigSin = np.sin((omega*x)/rate)
        self._rate = rate
        self._nbSamples = nbSamples
        self._data = fft.rfft(sigSin).__abs__()
        normData = np.linalg.norm(self._data)
        if normData != 0:
            self._data *= amp/np.linalg.norm(self._data)
        self._freq = freq
        self._freqs = fft.rfftfreq(2*nbSamples-1,d=1/rate)
        #self.plot()
    
    def getData(self):
        return self._data
    def getFreqMax(self):
        m = self.getData().argmax()
        return self._freqs[m]
    def plot(self):
        print('freq: ',self._freq, 'fm: ', self.getFreqMax())
        plt.clf()
        plt.plot(self._freqs, self._data)
        plt.show()
        

        
class FftStereoChunks:
    #_lChunks
    #_rChunks
    #_freqs
    def __init__(self, lenChunk, nbSamples, rate, left, right, freqs):
        self._lenChunk = lenChunk
        self._nbSamples = nbSamples
        self._rate = rate
        self._lChunks = left
        self._rChunks = right
        self._freqs = freqs
    
    def getChunkCount(self):
        return len(self._lChunks)
    
    def getSampleCount(self, index):
        return self._nbSamples[index]
    
    def print(self):
        print('-- FFT Stereo chunks --')
        print('Length of a chunk: ', self._lenChunk, ' s')
        print('Sample rate: ', self._rate,' Hz')
        print('Number of chunks', self.getChunkCount(), ' ')
        print('Number of samples in first chunk: ', self.getSampleCount(0), ' ')
        print('-- END FFT Stereo chunks --')
        
    
        
    def __init__(self, lenChunk, rate, data):
        print('Building FFT Stereo chunks...')
        self._lenChunk = lenChunk
        self._rate = rate
        #Duration of a sample = lenSample (in seconds)
        lenSample = 1/rate
        leftData, rightData = data[:,0], data[:,1]
        #number of chunks:
        nbChunks = int(len(leftData)/(lenChunk*rate))
        
        lChunksNotAveraged = array_split(leftData, nbChunks)
        rChunksNotAveraged = array_split(rightData, nbChunks)
        
        #
        #Computes FFT
        lChunksNotAveraged = array(list(map(fft.rfft, lChunksNotAveraged)))
        rChunksNotAveraged = array(list(map(fft.rfft, rChunksNotAveraged)))
        
        self._nbSamples = [len(lChunksNotAveraged[i]) for i in range(len(lChunksNotAveraged)) ]
        self._freqs = [fft.rfftfreq(2*self._nbSamples[i]-1,lenSample) for i in range(len(lChunksNotAveraged))]
       
        self._lChunks = lChunksNotAveraged
        self._rChunks = rChunksNotAveraged
        ####
        print('...Done')

        

def freqIsDuplicate(freq, oFreqs, delta=2**(1/12)):
    #delta as semitone in temperate scale
    for oFreq in oFreqs:
        if oFreq != 0 :
            #print('freq: ', freq, '  oFreq: ', oFreq, ' ratio: ', oFreq/freq)
            if freq < oFreq:
                if oFreq/freq < delta:
                    return True
            else:
                if freq/oFreq < delta:
                    return True
    return False
        
class ChunkRefactor :
    #
    # _threshold
    #
    def __init__(self, threshold):
        self._threshold = threshold
        
    def applyToChunk(self, freq, rate, nbSamples, n, lChunk, rChunk, lChunkAbs, rChunkAbs, indexMax, oFreqs, sValues):
        if not freqIsDuplicate(freq, oFreqs):
            #silly low-cut filter
            if freq > 20:
                oFreqs[n] = freq
                sValues[n] = (lChunk[indexMax], rChunk[indexMax])
            o = True
        else:
            o = False
        sinusoid =  FftSinusoid(freq, lChunkAbs[indexMax], rate, nbSamples)
        lChunkAbs -= sinusoid.getData()
        rChunkAbs -= sinusoid.getData()
        return o
        
        
    def applyToStereoChunk(self, rate, nbSamples, lChunk, rChunk, iFreqs):
        sValues = np.empty((nbSamples,2), dtype=np.dtype(np.complex64))
        oFreqs = np.zeros(nbSamples)
        #We retrieve the max value of the FFT
        #Get the freq, and then substract a Sin of the same freq
        #before going to the next one "threshold" times
        n = 0
        lChunk /= rate#*nbSamples
        rChunk /= rate#*nbSamples
        lChunkAbs = lChunk.__abs__()
        rChunkAbs = rChunk.__abs__()
        
        while n<self._threshold:
            indexMax = lChunkAbs.argmax()
            freq = iFreqs[indexMax]
            o = self.applyToChunk(freq, rate, nbSamples, n, lChunk, rChunk, lChunkAbs, rChunkAbs, indexMax, oFreqs, sValues)
            n+=o 
            indexMax = rChunkAbs.argmax()
            freq = iFreqs[indexMax]
            o = self.applyToChunk(freq, rate, nbSamples, n, lChunk, rChunk, lChunkAbs, rChunkAbs, indexMax, oFreqs, sValues)
            n+=o
        nonZeroN = np.count_nonzero(oFreqs)
        sValues.resize((nonZeroN,2))
        oFreqs.resize(nonZeroN)
        return sChunkX(oFreqs, sValues)
    
    def applyToFftStereoChunks(self, fftStereoChunks):
        nbChunks = fftStereoChunks.getChunkCount()
        nbSamples = fftStereoChunks._nbSamples
        freqs = fftStereoChunks._freqs
        rate = fftStereoChunks._rate
        sChunksX = nbChunks*[0]
        for i in range(nbChunks):
            chunk =  self.applyToStereoChunk(rate, nbSamples[i], fftStereoChunks._lChunks[i], fftStereoChunks._rChunks[i], freqs[i])
            sChunksX[i] = chunk
        return sChunksX
class sChunkX :
    #Freqency-indexed dictionnary -> (lValue, rValue) for each chunk
    #_chunkData
    def __init__(self, freqs, sValues):
        self._chunkData = dict()
        for i in range(len(freqs)):
            self._chunkData.update({freqs[i] : sValues[i]})
    def print(self):
        for freq in self._chunkData:
            print(freq,' Hz:', ' l -> ', self._chunkData[freq][0],' r -> ', self._chunkData[freq][1])
         
        
        
class RefStereoChunks:
    # _sChunksX
    # _lenChunk
    # _nbSamples
    # _rate
    def __init__(self, lenChunk, nbChunks, rate, sChunksX):
        self._lenChunk = lenChunk
        self._nbSamples
        self._rate = rate
        self._sChunksX =  sChunksX
        
    def __init__(self, fftStereoChunks, chunkRefactor):
        print('Refactoring...')
        self._lenChunk = fftStereoChunks._lenChunk
        self._nbSamples = fftStereoChunks._nbSamples
        self._rate = fftStereoChunks._rate
        self._sChunksX = chunkRefactor.applyToFftStereoChunks(fftStereoChunks)
        print('...Done')
    
    def getChunkCount(self):
        return len(self._sChunksX)
    
    def getRefactorCount(self, index):
        assert(index <= self.getChunkCount())
        return len(self._sChunksX[index]._chunkData)
        
    def print(self):
        print('-- REFACTORED Stereo chuncks --')
        print('Length of a chunk: ', self._lenChunk, ' s')
        print('Sample rate: ', self._rate,' Hz')
        print('Number of chunks: ', self.getChunkCount(), ' ')
        print('Number of refactors middle : ', self.getRefactorCount(self.getChunkCount()//2), ' ')
        #for sChunkX in self._sChunksX:
         #   sChunkX.print()
        print('-- END REFACTORED Stereo chunks --')
        
        
def listOfDicts(n):
    l = [0]*n
    for i in range(n):
        l[i] = dict()
    return l

class Point2D:
    #_x
    #_y
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def get(self):
        return (self._x, self._y)
    def print(self):
        print('(',self._x,self._y,')',end='')

def position1(lC, rC):
    if np.abs(lC) == 0:
        x = -0.5
        y = 0
    elif np.abs(rC) == 0 :
        x= 0.5
        y = 0
    else:
        x = np.arctan( m.log( (np.abs(lC)/np.abs(rC))**3 ) )/np.pi
        y = np.arctan(( m.log( ((np.abs(np.real((rC))) + np.abs(np.real(lC)))/(np.abs(np.imag(rC)) + np.abs(np.imag(lC))))**3 )  ))/np.pi
    return Point2D(x,y)

def posFunctor(sC):
    return position1(sC[0], sC[1])

    

class TransClassifier:
    # lC, rC : left, right coeffs for same freq
    # _posFunctor( complex lC, rC ) -> [-1, 1]
    # _pMaxX
    # _pMaxY
    def __init__(self, posFunctor):
        self._posFunctor = posFunctor
        
class PositionClassifier:
    # _posFunctor
    def __init__(self, posFunctor):
        self._posFunctor = posFunctor
    def classify(self, sChunkF):
        sChunkPos = listOfDicts(len(sChunkF))
        for i in range(len(sChunkF)):
            for freq, sValue in sChunkF[i].items():
                sChunkPos[i].update({freq : self._posFunctor(sValue)})
        return sChunkPos
    
class Space:
    # _freqStart
    # _freqEnd
    # _interName
    # _transName
    def __init__(self, freqStart, freqEnd, interpolate, transform):
        self._freqStart = freqStart
        self._freqEnd = freqEnd
        self._interName = interpolate
        self._transName = transform
    def print(self):
        print('Fstart: ', self._freqStart, 'Fend: ', self._freqEnd, 'interName: ', self._interName, 'transName: ', self._transName)
        
def spacePresFunctor(freq1, freq2):
    dist = abs(freq1-freq2)/max(freq2, freq1)
    normal = 2/np.pi*np.arctan(1/dist)
    return normal
    
class SpaceClassifier:
    # _spaceEnds [space0._freqEnd, space2._freqEnd, ..., space(n-1)._freqEnd]
    # _nbSpaces
    # _presFunctor
    def __init__(self, spaceEnds, presFunctor):
        self._spaceEnds = spaceEnds
        self._nbSpaces = len(spaceEnds)
        self._presFunctor = presFunctor
    def __init__(self, listSpaces, presFunctor):
        self._nbSpaces = len(listSpaces)
        self._presFunctor = presFunctor
        self._spaceEnds = [0]*self._nbSpaces
        for i in range(len(listSpaces)):
            self._spaceEnds[i] = listSpaces[i]._freqEnd
        
    def classifyFreq(self, freq):
        centre = [0]*self._nbSpaces
        for i in range(1,self._nbSpaces):
            centre[i] = (self._spaceEnds[i] - self._spaceEnds[i-1])/2
            if freq < self._spaceEnds[i]:
                #presence1 = self._presFunctor(centre[i-1], freq)
                #presence2 = self._presFunctor(centre[i], freq)
                presence1 = 0
                presence2 = 1
                return i, presence1, presence2
        return i, presence1, presence2
        #
            
    def classify(self, sChunkX):
        sChunkA = listOfDicts(self._nbSpaces)
        sChunkPres = listOfDicts(self._nbSpaces)
        sChunkF = listOfDicts(self._nbSpaces)
        for freq, sValue in sChunkX._chunkData.items():
            amp = np.linalg.norm(sValue)
            indexSpace, pres1, pres2 = self.classifyFreq(freq)
            sChunkF[indexSpace-1].update({freq : sValue})
            sChunkF[indexSpace].update({freq : sValue})
            sChunkA[indexSpace-1].update({freq : amp})
            sChunkA[indexSpace].update({freq : amp})
            sChunkPres[indexSpace-1].update({freq : pres1})
            sChunkPres[indexSpace].update({freq : pres2})
        return sChunkA, sChunkPres, sChunkF
    
    
    
def restDistance1(pos1, pos2):
    d  = (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2
    return d

def restPointTooClose(restPos, xList, yList, pointRange, distMin, restPosFunctor):
    for i in range(len(xList)):
        pos = [xList[i],yList[i]]
        if restPosFunctor(pos, restPos) < distMin :            
            return True
    return False

def addRestPoints(xList, yList, ampList, nbRestPoints=32, pointRange=1, distMin=0.03, restPosFunctor=restDistance1):
    #X and Y axis centred around 0
    for xR in range(nbRestPoints):
        for yR in range(nbRestPoints):
            restPos = [0]*2
            restPos[0] = (pointRange/nbRestPoints)*(xR - pointRange/2)
            restPos[1] = (pointRange/nbRestPoints)*(yR - pointRange/2)
            if not restPointTooClose(restPos, xList, yList, pointRange, distMin, restPosFunctor):
                xList.append(restPos[0])
                yList.append(restPos[1])
                ampList.append(0)

class sChunkSpace:
    # _listSpaces
    # _sChunkA[... nbSpaces]
    # _sChunkPos[...nbSpaces]
    # _sChunkPres[...nbSpaces]
    # _Rbf[...nbSpaces]
    def __init__(self, sChunkX, listSpaces, spaceC, positionC):
        self._sChunkA, self._sChunkPres, sChunkF = spaceC.classify(sChunkX)
        self._sChunkPos = positionC.classify(sChunkF)
        self._listSpaces = listSpaces
        self._spaceC = spaceC
        self._positionC = positionC
        #classify space
        #classify position

    def initInterpolation(self):
        self._Rbf = []
        for s in range(len(self._listSpaces)):
            xList = []
            yList = []
            ampList = []
            for freq, pos in self._sChunkPos[s].items():
                x = pos._x
                y = pos._y
                xList.append(x)
                yList.append(y)
                ampList.append(self._sChunkA[s][freq])
            addRestPoints(xList, yList, ampList)
            #WON'T WORK WITH ONLY ONE POINT OF DATA
            #print('x:',*xList)
            #print('y: ',*yList)
            #print('amp: ', *ampList)
            rbf  = ip.Rbf(xList,yList, ampList, function=self._listSpaces[s]._interName)
            #print('Ahhhh', type(rbf))
            self._Rbf.append(rbf)
                

            
    def print(self):
        for s in range(len(self._listSpaces)):
            self._listSpaces[s].print()
            for freq in self._sChunkA[s]:
                print('freq: ',freq, 'amp:',self._sChunkA[s][freq],' pos: ',end='')
                self._sChunkPos[s][freq].print()
                print(' presence: ', self._sChunkPres[s][freq])
         

    

class SignalSpace:
    # _sChunksS :[...nbChunks]
    # _listSpaces
    # _lenChunk
    # _nbSamples
    # _rate
    def __init__(self, refStereoChunks, listSpaces, spaceC, positionC):
        self._lenChunk = refStereoChunks._lenChunk
        self._nbSamples = refStereoChunks._nbSamples
        self._rate = refStereoChunks._rate
        self._listSpaces = listSpaces
        self._sChunksS = [0]*refStereoChunks.getChunkCount()
        for i in range(len(self._sChunksS)):
            print('*', flush=True,sep='', end='')
            self._sChunksS[i] = sChunkSpace(refStereoChunks._sChunksX[i], listSpaces, spaceC, positionC)
        
        for s in range(len(self._listSpaces)):
                self.normaliseSpaceMaxAmp(s)
        for i in range(len(self._sChunksS)):
            self._sChunksS[i].initInterpolation()
                
        print()
        
    def normaliseSpaceMaxAmp(self, iSpace):
        nbChunks = len(self._sChunksS)
        sChunksA = [0]*nbChunks
        absMax = 0
        for k in range(nbChunks):
            sChunksA[k] = self._sChunksS[k]._sChunkA[iSpace]
            if sChunksA[k]:
                currentMax = max(sChunksA[k])
                if currentMax > absMax:
                    absMax = currentMax
        for k in range(nbChunks):
            for freq, amp in self._sChunksS[k]._sChunkA[iSpace].items():
                if absMax != 0:
                    self._sChunksS[k]._sChunkA[iSpace][freq] *= 1/270#*1/absMax
        
    
            
    def getRbfs(self):
        nbSpaces = len(self._listSpaces)
        Rbfs = [[] for j in range(nbSpaces)]
        for i in range(nbSpaces):
            for k in range(len(self._sChunksS)):
                Rbfs[i].append(self._sChunksS[k]._Rbf[i])
        return Rbfs
                    
            

    def print(self):
        print('-- SIGNAL SPACE --')
        print('Length of a chunk: ', self._lenChunk, ' s')
        print('Sample rate: ', self._rate,' Hz')
        print('Spaces: ')
        for space in self._listSpaces:
            space.print()
        print('Middle sChunkS: ')
        self._sChunksS[len(self._sChunksS)//2].print()
        print('-- END SIGNAL SPACE --')  
