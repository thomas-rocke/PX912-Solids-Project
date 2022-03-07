from cv2 import solve
import matplotlib
from FEMSolver import FEM
from NodeDists import *
from Meshing import Mesh
import numpy as np
import matplotlib.pyplot as plt

bot = [0, 0.5, 1, 1.5, 2]                     # x-coordinates of bottom side nodes
top = [0, 0.5, 1, 1.5, 2]                     # x-coordinates of top side nodes
left = [0, 0.5, 1]                            # y-coordinates of left-hand side nodes 
right = [0.5, 0.75, 1]                        # y-coordinates nodes of right-hand side nodes

corners = np.array([[0, 1], # top left
                    [2, 1], # top right
                    [2, 0.5], # bottom right
                    [0, 0]]) # bottom left
'''
plt.scatter(*CornerBias(corners, 20, 20), color='k')
plt.scatter(corners[:, 0], corners[:, 1], color='r')
plt.show()
'''

n = 10

mesh = Mesh(corners, n, n, coord_func=Uniform)
'''
corners2 = np.array([[0, 2],
                     [3, 2],
                     [2, 1],
                     [0, 1]])
mesh2 = Mesh(corners2, n, n, coord_func=Uniform)
solver2 = FEM(mesh2, 1, 1)
solver2.pin_edge('bottom', 1)

#mesh.plot()
#mesh2.plot()

comb = mesh + mesh2
comb.plot(show_ids=True)

'''
force = np.array([0, 20])
edge='top'

solver = FEM(mesh, 3E7, 0.3, quad_points=5)
mesh.apply_load(force, edge)
mesh.pin_edge('left', 0)
mesh.pin_edge('left', 1)


mesh.plot(show_ids=False)
solver.solve()
solver.show_deformation(magnification=10000)
solver.get_props()
solver.plot_props()
plt.show()
