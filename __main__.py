import sys, os
import bpy
import scipy.io.wavfile as wv
import scipy.interpolate as ip


dir = os.path.dirname(bpy.data.filepath)+'/source'
musicDir = '/audio/'
print(dir)

if not dir in sys.path:
    sys.path.append(dir)

#change directory
os.chdir(dir)

import script_music as sm
import script_animation as an


def main(argv) :
    if len(argv) != 6 :
        return
    #Data as a numpy array (SECOND DIMENSION IF STEREO LEFT/RIGHT
    lenChunk = argv[0]
    fftThreshold = argv[1]
    listSpaces = argv[2]
    spaceC = argv[3]
    positionC = argv[4]
    rate, data =  wv.read(argv[5])
    #
    transform = sm.FftStereoChunks(lenChunk,  rate, data)
    transform.print()
    refactor = sm.ChunkRefactor(fftThreshold)
    transform = sm.RefStereoChunks(transform, refactor)
    transform.print()
    signalSpace = sm.SignalSpace(transform, listSpaces, spaceC, positionC)
    signalSpace.print()
    
    worldAnimation = an.WorldAnimation(signalSpace)
    worldAnimation.run()
    return

if __name__ == "__main__":
        #from the script directory
        track = dir+musicDir+'long_mountain.wav'
        #in seconds
        lenChunk = 0.5
        #in what exactly?
        fftThreshold = 10
        #track = sys.argv[:1]
        space1 = sm.Space(0,220, 'gaussian', 'translate')
        space2 = sm.Space(220,880, 'gaussian','translate')
        space3 = sm.Space(880,1760, 'thin-plate', 'translate')
        space4 = sm.Space(1760, 22050, 'inverse', 'translate')
        listSpaces = [space1, space2, space3, space4]
        spaceC = sm.SpaceClassifier(listSpaces, sm.spacePresFunctor)
        positionC = sm.PositionClassifier(sm.posFunctor)
        main([lenChunk, fftThreshold, listSpaces, spaceC, positionC, track])
