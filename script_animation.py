import bpy
import sys
import bmesh
import numpy as np


from numpy import array, empty

def translateZ(v, amount):
    v.co.z += abs(amount)
    pass

class MeshTransform:
    # _bMesh
    # _meshSize
    # _listPoints
    # _listAmps
    # _transFunctor
    # _shapeKeyBasis
    # _Rbf
    
    def __init__(self, mesh, meshSize, space):
        self._transFunctor = FunctorTransform(space._transName)
        #
        data = mesh.data
        self.assignObject(data)
        self._meshSize = meshSize
        #select the object
        bpy.ops.object.select_pattern(pattern=mesh.name)
        #self.getListPointsFromVertices()
        bpy.ops.object.select_all(action="DESELECT")
        #BASIS
        self._shapeKeyBasis = mesh.shape_key_add(name="Basis", from_mix=False)
        #
        
    def assignObject(self, activeObject):
        self._bMesh = bmesh.new()
        self._bMesh.from_mesh(activeObject)
        
    def getListPointsFromVertices(self):
        self._listPoints = ([],[],[])
        for v in self._bMesh.verts:
            self._listPoints[0].append(v.co.x)
            self._listPoints[1].append(v.co.y)
            self._listPoints[2].append(v.co.z)
      
    #WON'T HAVE TO USE, WILL WORK DIRECTLY ON V!!  
    #IN FACT WILL USE TO RESET VERTICES AFTER UPDATING
    def setVerticesFromListPoints(self):
        i = 0
        #self._bMesh.verts = self._shapeKeyBasis.data
        for v in self._bMesh.verts:
            #v.location = self._listPoints[:][i]
            v.co.x = self._listPoints[0][i]
            v.co.y = self._listPoints[1][i]
            v.co.z = self._listPoints[2][i]
            i+=1
    #EACH TIME AFTER UPDATE
    def resetVertices(self):
        self.setVerticesFromListPoints()
            
    #the only thing which changes from one frame to the next is the Rbf
    #that is, the control points thereof
    def update(self, Rbf):
        self._Rbf = Rbf
        #self.interpolateAmps()
    
    #def interpolateAmps(self):
    #    self._listAmps = self._Rbf(self._listPoints[0], self._listPoints[1])
        
    def applyVertice(self, v):
        amp = self._Rbf(v.co.x, v.co.y)
        self._transFunctor.apply(v, amp)
        
    def apply(self, currentShapeKey):
        for v in currentShapeKey.data:
            self.applyVertice(v)
            

        
        
        

class FunctorTransform:
    #_transformType : 'translate', 'scale',....
    #_function
    def __init__(self, transformType):
        if transformType == 'translate':
            self._function = translateZ
        #elif transformType == 'scale':
        #    self._function = bmesh.ops.scale
        #elif transformType == 'rotate':
        #    self._function = bmesh.ops.rotate
        self._transformType = transformType
    def apply(self, v, amp):
        self._function(v,amp)
        
class WorldAnimation:
    #_keyingSets ???
    # _lenChunk
    # _fps
    # _transforms[...nbSpaces]
    # _Rbfs[...nbSpaces][...nbChunks]
    #listMeshes[...nbSpaces]
    # _listSpaces
        
    def __init__(self,  signalSpace):
        self._Rbfs = signalSpace.getRbfs()
        self._nbChunks = len(self._Rbfs[0])
        self._lenChunk = signalSpace._lenChunk
        self._listSpaces = signalSpace._listSpaces
        #
        #self._keyingSet = bpy.ops.anim.keying_set_add()
        self._fps = bpy.context.scene.render.fps*bpy.context.scene.render.fps_base
        self._frameStart = 0
        self._frameEnd = self._lenChunk*self._nbChunks*self._fps
        self._keyTimes = []
        self._keyFrames = []
        #
        self._listMeshes = []
        self._transforms = [0]*len(self._listSpaces)
        #
        self.clearScene()
        #
        
        meshSize = 1.5
        for i in range(len(self._listSpaces)):
            self.addMesh(meshSize, str(i))
            self._transforms[i] = MeshTransform(self._listMeshes[i], meshSize, self._listSpaces[i])
        
    def clearScene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    
    def addKeyTransform(self, iSpace, transform, time):
        #bpy.ops.object.select_pattern(pattern=self._listMeshes[iSpace].name)
        bpy.context.scene.frame_set(frame=time)
        mesh = self._listMeshes[iSpace]
        #
        currentShapeKey = mesh.shape_key_add(from_mix = False)
        #currentShapeKey.add_key_frame]
        transform.apply(currentShapeKey)
        #
        #bpy.ops.object.select_all(action="DESELECT")
        #
        self._keyTimes.append(time)
        #transform.resetVertices()
        
    def addMesh(self, meshSize, name):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.mesh.primitive_grid_add(size=meshSize, x_subdivisions = 20, y_subdivisions = 20, enter_editmode=False, location=(0, 0, 0))
        mesh = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active= mesh
        #
        mesh.name = name
        mesh.data.name = name
        #
        self._listMeshes.append(mesh)
        
        
    def run(self):
        nbChunks = self._nbChunks
        nbSpaces = len(self._listSpaces)
        #bpy.context.scene.frame_start = 0
        #bpy.context.scene.frame_end = nbChunks
        for i in range(nbSpaces):
            print("Space #",i)
            mesh = self._listMeshes[i]
            bpy.context.scene.frame_set(self._frameStart)
            mesh.keyframe_insert(data_path="location")
            for k in range(nbChunks):
                frame = k*self._lenChunk*self._fps
                self._transforms[i].update(self._Rbfs[i][k])
                self.addKeyTransform(i, self._transforms[i], frame)
                #
                bpy.data.shape_keys["Key"].use_relative = False
                #
            bpy.context.scene.frame_set(self._frameEnd)
            mesh.keyframe_insert(data_path="location")
            