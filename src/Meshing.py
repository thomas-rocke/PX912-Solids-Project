import matplotlib.pyplot as plt
import numpy as np

from NodeDists import *

class Mesh():
    '''
    Meshing class
    
    Uses modified code from the given PX912 mesh generation functions, 
    and is heavily inspired by the PX920 workshop 1 Mesh class 

    '''
    def __init__(self, corners, nx, ny, coord_func=Uniform):
        '''
        Inits Mesh
        corners is a shape (4, 2) array of coords
        [0, :] is top left, [1, :] top right
        [2, :] bottom right, [3, :] bottom left

        coord_func(corners, nx, ny) is a generates the coordinates for each of the integration points
        '''
        self.corners=corners
        self.nx = nx
        self.ny = ny
        self.nnodes = nx * ny
        self.node_dist = coord_func
        self.make_mesh()
        self.pins = np.array([False] * 2 * nx * ny).reshape((nx*ny, 2))

        self.edges = {'top' : np.array([ny * (i + 1) - 1 for i in range(nx)]),
                      'bottom' : np.array([i * (ny) for i in range(nx)]),
                      'left' : np.array([i for i in range(ny)]),
                      'right' : np.array([(nx-1)*ny + i for i in range(ny)])}
    
    def make_mesh(self):
        '''
        Generate mesh from corners and nx, ny
        Sets self.XY to integration point coords
        with shape (2, nx * ny)

        self.ELS is shape ((nx-1) * (ny-1), 4), and gives the 4 integration points corresponding tro each node

        self.DOF is shape ((nx-1) * (ny-1), 4) and gives DOF for each element
        '''
        self.XY = np.array(self.node_dist(self.corners, self.nx, self.ny))
        nelx = self.nx - 1
        nely = self.ny - 1
        nnodes = nelx*nely
        self.ELS = np.zeros((nnodes, 4), dtype=int)

        for i in range(nelx):
            for j in range(nely):
                self.ELS[j+i*nelx, :] = [j+i*self.nx, j+i*self.nx+1,j+(i+1)*self.nx+1, j+(i+1)*self.nx]

        self.DOF = np.zeros((nnodes, 8), dtype=int)

        for i in range(nnodes):
            self.DOF[i, :] = [self.ELS[i,0]*2, self.ELS[i,1]*2-1, self.ELS[i,1]*2, self.ELS[i,1]*2+1, self.ELS[i,2]*2, self.ELS[i,2]*2+1, self.ELS[i,3]*2, self.ELS[i,3]*2+1]


    def plot(self, show_ids = True):
        '''
        Plots the constructed mesh
        '''
        title = "Mesh"

        #Fully Pinned
        cond = self.pins[:, 0] * self.pins[:, 1]
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sg', label='Fully Pinned')

        #Pinned in x1 only
        cond = self.pins[:, 0] * (1 - self.pins[:, 1])
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sr', label='Pinned in x1')
        #Pinned in x2 only
        cond = (1 - self.pins[:, 0]) * self.pins[:, 1]
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sb', label='Pinned in x2')
        #Free
        cond = 1 - (self.pins[:, 0] + self.pins[:, 1])
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sk', label='Free')

        for i in range(len(self.ELS)):
            plt.fill(self.XY[self.ELS[i, :], 0], self.XY[self.ELS[i, :], 1], edgecolor='k', fill=False)


        if show_ids:
            for i in range(4):                             #loop over all nodes within an element
                for j in range(len(self.ELS)):                  #loop over all elements
                    sh=0.01
                    plt.text(self.XY[self.ELS[j,i],0]+sh,self.XY[self.ELS[j,i],1]+sh, self.ELS[j,i])

        # Set chart title.
        plt.title(title, fontsize=19)
        # Set x axis label.
        plt.xlabel("$x_1$", fontsize=10)
        # Set y axis label.
        plt.ylabel("$x_2$", fontsize=10)

        plt.legend()

        plt.show()

    def set_pins(self, pins):
        '''
        Defines pinning conditions

        pins should be a shape (nnodes, 2) boolean array
        pins[i, j] pins node i in the xj direction
        '''
        self.pins = pins
        
