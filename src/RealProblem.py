import numpy as np
from Meshing import Mesh
from NodeDists import Uniform, CornerBias
from FEMSolver import FEM

n = 10
func = Uniform
E = 80.0E9
nu = 0.3



corners1 = np.array([[0, 0.1],
                     [0.1, 0.1],
                     [0.1, 0],
                     [0, 0]])
mesh1 = Mesh(corners1, n, n, func)
mesh1.pin_edge('bottom', 0)
mesh1.pin_edge('bottom', 1)
mesh1.pin_edge('left', 0)

corners2 = np.array([[0.1, 0.1], 
                     [0.25, 0.1],
                     [0.25, 0],
                     [0.1, 0]])
mesh2 = Mesh(corners2, n, n, func)
mesh2.pin_edge('bottom', 0)
mesh2.pin_edge('bottom', 1)

corners3 = np.array([[0, 0.6],
                     [0.1, 0.6],
                     [0.1, 0.1],
                     [0, 0.1]])
mesh3 = Mesh(corners3, n, n, func)
mesh3.pin_edge('left', 0)

corners4 = np.array([[0, 0.7], 
                     [0.1, 0.7],
                     [0.1, 0.6],
                     [0, 0.6]])
mesh4 = Mesh(corners4, n, n, func)
mesh4.pin_edge('left', 0)
mesh4.apply_load([0, 100], 'top')

corners5 = np.array([[0.1, 0.7],
                     [0.25, 0.7],
                     [0.25, 0.6],
                     [0.1, 0.6]])
mesh5 = Mesh(corners5, n, n, func)
mesh5.apply_load([0, 100], 'top')

shape_outline = np.array([[0, 0],
                          [0, 0.7],
                          [0.25, 0.7],
                          [0.25, 0.6],
                          [0.1, 0.6],
                          [0.1, 0.1],
                          [0.25, 0.1],
                          [0.25, 0],
                          [0, 0]])



#mesh4.plot()

#print(np.sum(mesh4.pins))

s4 = mesh4 + mesh5

#print(np.sum(s4.pins))
s3 = mesh3 + s4
s2 = mesh1 + s3
shape = mesh2 + s2
#shape.plot(show_ids=False)


solver = FEM(shape, E, nu)

solver.solve()
#solver.show_deformation(10000)
solver.get_props(10)
solver.plot_props(shape_outline=shape_outline)