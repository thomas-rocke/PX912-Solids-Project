'''
Solves FEM given mesh and material properties

'''
from cProfile import label
from attr import attributes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyrsistent import v
from tqdm import tqdm
from copy import deepcopy as copy

gauss_eval_points = { # Evaluation coordinates for Gaussian Quadrature integration
    1 : np.array([0]),
    2 : np.array([-np.sqrt(1/3), np.sqrt(1/3)]),
    3 : np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]),
    4 : np.array([-np.sqrt(3/7 + 2 * np.sqrt(6/5)/7), -np.sqrt(3/7 - 2 * np.sqrt(6/5)/7), np.sqrt(3/7 - 2 * np.sqrt(6/5)/7), np.sqrt(3/7 + 2 * np.sqrt(6/5)/7)]),
    5 : np.array([-np.sqrt(5 + 2 * np.sqrt(10/7))/3, -np.sqrt(5 - 2 * np.sqrt(10/7))/3, 0, np.sqrt(5 - 2 * np.sqrt(10/7))/3, np.sqrt(5 + 2 * np.sqrt(10/7))/3])
}

gauss_weights = { # Weights for Gaussian Quadrature integration
    1 : np.array([2]),
    2 : np.array([1, 1]),
    3 : np.array([5/9, 8/9, 5/9]),
    4 : np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36]),
    5 : np.array([(322 - 13 * np.sqrt(70))/900, (322 + 13 * np.sqrt(70))/900, 128/225, (322 + 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900])
}


import numpy as np
from Meshing import Mesh

def Plane_Stress(x, y, E, nu):
    '''C matrix in the case of plane stress'''
    const = E/(1.0-nu*nu)
    C = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5*(1.0-nu)]])
    return const*C

def Plane_Strain(x, y, E, nu):
    '''C matrix in the case of plane strain'''
    const = E/(1.0-nu)/(1 + nu)
    C = np.array([[1.0 - nu, nu, 0.0], [nu, 1.0-nu, 0.0], [0.0, 0.0, 0.5*(1.0-2 * nu)]])
    return const*C

def xi_from_x(x):
    '''
    Find xi from x coord
    '''
    return x - 1.0


def eta_from_y_xi(y, xi):
    '''
    Find eta from y coord and xi
    '''
    return (8 * y - 5)/(3 - xi)



def N(xi, eta):
    '''Nodal functions; returns a vector as a function of xi, eta'''
    N1 = 0.25*(1.0-xi)*(1.0+eta)
    N2 = 0.25*(1.0-xi)*(1.0-eta)
    N3 = 0.25*(1.0+xi)*(1.0-eta)
    N4 = 0.25*(1.0+xi)*(1.0+eta)
    return np.array([N1, N2, N3, N4])


def J(xi, eta):
    '''Nodal Jacobian as a function of xi, eta'''
    return np.array([[1.0, 0.0],[0.125 - 0.125*eta, 0.375 - xi*0.125]])

def J_inv(xi, eta):
    '''Inverse Nodal Jacobian'''
    Jac = J(xi, eta)
    return np.linalg.inv(Jac)


class FEM():
    def __init__(self, mesh, E, nu, elasticity_func = Plane_Stress, quad_points=2):
        '''
        Initialise FEM solver
        '''

        self.mesh = mesh 
        self.E = E
        self.nu = nu
        self.elasticity = elasticity_func
        self.quad_points = quad_points

        self.nnodes = self.mesh.XY.shape[0]
        self.K = np.zeros((2 * self.nnodes, 2 * self.nnodes))
        self.displacements = np.zeros((2 * self.nnodes))
        self.stresses = None
        self.strains = None

    @property
    def forces(self):
        '''
        Alias for forces
        '''
        return self.mesh.forces


    def strain_displacement(self, corners, xi, eta):
        '''
        Find the strain-displacement matrix for the element given by corners, xi, eta
        '''

        nat_coords = np.array([[-1, 1, 1, -1],[-1, -1, 1, 1]]) # Natural (square) coord system
        
        dN = np.zeros((2,4)) # Gradient of Shape functions
        dN[0,:]=(1/4)*nat_coords[0,:]*(1+nat_coords[1,:]*eta)
        dN[1,:]=(1/4)*nat_coords[1,:]*(1+nat_coords[0,:]*xi)

        # Real-space Jacobian
        Jmat = dN @ corners
        Jmat_inv = np.linalg.inv(Jmat)

        dNdx = np.dot(Jmat_inv, dN)

        # Strain-displacement matrix
        B = np.zeros((3,8))
        B[0,0::2] = dNdx[0,:]
        B[1,1::2] = dNdx[1,:]
        B[2,0::2] = dNdx[1,:]
        B[2,1::2] = dNdx[0,:]
        
        return B
    
    def Gauss_quad(self, corners):
        '''
        Perform Gauss Quadrature integration using self.quad_points**2 total points
        '''
        points = gauss_eval_points[self.quad_points]
        weights = gauss_weights[self.quad_points]
        k_element = np.zeros((8, 8))

        C = self.elasticity(*corners[0, :], self.E, self.nu)
        for i in range(self.quad_points):
            for j in range(self.quad_points):
                xi = points[i]
                eta = points[j]

                # Strain-displacement matrix
                B = self.strain_displacement(corners, xi, eta)

                # Construct elemental k
                k_element += (B.T @ C @ B) * weights[i] * weights[j]
        return k_element     

    def eval_K(self):
        '''
        Evaluate stiffness matrix for all integration points
        '''
        # Reset K to 0
        self.K = np.zeros((2 * self.nnodes, 2 * self.nnodes))
        for i in tqdm(range(self.mesh.ELS.shape[0]), desc="Evaluating elemental Ks"):
            element = self.mesh.ELS[i, :]
            # Find local k matrix
            corners = self.mesh.XY[element, :]
            k_element = self.Gauss_quad(corners)

            # Add local matrices together
            self.K[np.ix_(self.mesh.DOF[i, :], self.mesh.DOF[i, :])] += k_element

    def solve(self):
        '''
        Solves the FEM problem given by the mesh, pinning, loading, and material properties
        '''
        # Compute K matrix
        self.eval_K()
        # Separate the K matrix out into components
        is_pinned = self.mesh.pins.flatten()
        K_ee = self.K[is_pinned == False, :][:, is_pinned == False]
        K_ef = self.K[is_pinned == False, :][:, is_pinned == True]
        K_ff = self.K[is_pinned == True, :][:, is_pinned == True]

        Forces = self.mesh.forces.flatten()[is_pinned == False]
        
        disps = np.linalg.solve(K_ee, Forces)

        reactions = K_ef.T @ disps
        self.displacements[is_pinned==False] = disps
        self.total_force = (self.K @ self.displacements)

    def show_deformation(self, magnification=1):

        disps = magnification * self.displacements

        old_XY = self.mesh.XY
        new_XY = self.mesh.XY + disps.reshape((self.nnodes, 2))
        plt.plot(old_XY[:, 0], old_XY[:, 1], 'sk', label='Undeformed shape')
        plt.plot(new_XY[:, 0], new_XY[:, 1], 'sr', label='Deformed Shape')

        for i in range(len(self.mesh.ELS)):
            plt.fill(old_XY[self.mesh.ELS[i, :], 0], old_XY[self.mesh.ELS[i, :], 1], edgecolor='k', fill=False)
            plt.fill(new_XY[self.mesh.ELS[i, :], 0], new_XY[self.mesh.ELS[i, :], 1], edgecolor='r', fill=False)
        # Set chart title.
        plt.title("Mesh Deformation under loading", fontsize=19)
        # Set x axis label.
        plt.xlabel("$x_1$", fontsize=10)
        # Set y axis label.
        plt.ylabel("$x_2$", fontsize=10)

        plt.legend()

        plt.show()
    
    def get_props(self, lin_samples_per_elem:int=15):
        '''
        Get strain of solid, evaluating lin_samples_per_elem**2 points for each element
        '''
        nels = self.mesh.ELS.shape[0]

        self.coords = np.zeros((nels*(lin_samples_per_elem)**2, 2)) #(x, y) coords for each evaluation point
        self.strains = np.zeros((nels*(lin_samples_per_elem)**2, 3)) # strains for each evaluation point
        self.stresses = np.zeros_like((self.strains))
        self.disp_field = np.zeros_like((self.coords))
        self.force_field = np.zeros_like((self.coords))


        xis = np.linspace(-1, 1, lin_samples_per_elem)
        etas = np.linspace(-1, 1, lin_samples_per_elem)

        xis, etas = np.meshgrid(xis, etas)

        for i in tqdm(range(self.mesh.ELS.shape[0]), desc="Generating stress coords across elements"):
            el = self.mesh.ELS[i, :]
            d = self.displacements[self.mesh.DOF[i, :]]
            forces = self.total_force[self.mesh.DOF[i, :]]
            corners = self.mesh.XY[el]
            c = self.elasticity(*corners[0, :], self.E, self.nu)

            for j in range(lin_samples_per_elem):
                for k in range(lin_samples_per_elem):
                    idx = i * lin_samples_per_elem**2  + j * lin_samples_per_elem + k

                    b = self.strain_displacement(corners, xis[j, k], etas[j, k])
                    shapes = N(xis[j, k], etas[j, k])
                    self.strains[idx] = b @ d
                    self.coords[idx] = shapes @ corners
                    self.stresses[idx] =  c @ self.strains[idx]
                    self.disp_field[idx] = shapes @ d.reshape((4, 2))
                    self.force_field[idx] = shapes @ forces.reshape((4, 2))


    def plot_props(self, shape_outline=None):
        def mask_outside_polygon(poly_verts, ax=None):
            """
            Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
            all areas outside of the polygon specified by "poly_verts" are masked.  

            "poly_verts" must be a list of tuples of the verticies in the polygon in
            counter-clockwise order.

            Returns the matplotlib.patches.PathPatch instance plotted on the figure.
            """
            import matplotlib.patches as mpatches
            import matplotlib.path as mpath

            if ax is None:
                ax = plt.gca()

            # Get current plot limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Verticies of the plot boundaries in clockwise order
            outside_vertices = np.array([[xlim[0], ylim[0]],
                                         [xlim[0], ylim[1]],
                                         [xlim[1], ylim[1]],
                                         [xlim[1], ylim[0]],
                                         [xlim[0], ylim[0]]])[:, :, None]

            outside_vertices = np.hstack((outside_vertices[:, 0, :], outside_vertices[:, 1, :]))


            inside_vertices = np.hstack((poly_verts[:, 0][:, None], poly_verts[:, 1][:, None]))

            ins_codes = np.ones(
                len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
            ins_codes[0] = mpath.Path.MOVETO

            ots_codes = np.ones(
                len(outside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
            ots_codes[0] = mpath.Path.MOVETO

            # Concatenate the inside and outside subpaths together, changing their
            # order as needed
            vertices = np.concatenate((outside_vertices[::1],
                                    inside_vertices[::-1]))
            # Shift the path
            #vertices[:, 0] += i * 2.5
            # The codes will be all "LINETO" commands, except for "MOVETO"s at the
            # beginning of each subpath
            all_codes = np.concatenate((ots_codes, ins_codes))
            # Create the Path object
            path = mpath.Path(vertices, all_codes)
            # Add plot it
            patch = mpatches.PathPatch(path, facecolor='white', edgecolor='black')
            ax.add_patch(patch)

            return patch

        def plotting_3(x, y, data, labels, corners):
            fig, ax = plt.subplots(nrows=2, ncols = 2)

            #poly2 = mask_outside_polygon(corners, ax[1, 0])
            #poly3 = mask_outside_polygon(corners, ax[0, 1])
            #poly4 = mask_outside_polygon(corners, ax[1, 1])
            vmax = np.max(np.abs(data))
            vmin = -vmax

            col = ax[0, 0].tripcolor(x, y, data[:, 0], shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[0, 0].set_title(labels[0])
            mask_outside_polygon(corners, ax[0, 0])


            ax[1, 0].tripcolor(x, y, data[:, 1], shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[1, 0].set_title(labels[1])
            mask_outside_polygon(corners, ax[1, 0])

            ax[0, 1].tripcolor(x, y, data[:, 2], shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[0, 1].set_title(labels[2])
            mask_outside_polygon(corners, ax[0, 1])

            ax[1, 1].tripcolor(x, y, np.linalg.norm(data, axis=1), shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[1, 1].set_title(labels[3])
            mask_outside_polygon(corners, ax[1, 1])
            
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(col, cax=cbar_ax)
            plt.show()

        def plotting_2(x, y, data, labels, corners):
            fig, ax = plt.subplots(nrows=2, ncols = 2)

            #poly2 = mask_outside_polygon(corners, ax[1, 0])
            #poly3 = mask_outside_polygon(corners, ax[0, 1])
            #poly4 = mask_outside_polygon(corners, ax[1, 1])

            vmax = np.max(np.abs(data))
            vmin = -vmax

            col = ax[0, 0].tripcolor(x, y, data[:, 0], shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[0, 0].set_title(labels[0])
            mask_outside_polygon(corners, ax[0, 0])


            ax[1, 0].tripcolor(x, y, data[:, 1], shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[1, 0].set_title(labels[1])
            mask_outside_polygon(corners, ax[1, 0])

            ax[0, 1].tripcolor(x, y, np.linalg.norm(data, axis=1), shading='gouraud', vmin=vmin, vmax=vmax, clip_on=True, cmap='bwr')
            ax[0, 1].set_title(labels[2])
            mask_outside_polygon(corners, ax[0, 1])
            
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(col, cax=cbar_ax)
            plt.show()

        
        coords = self.coords
        strains = self.strains
        stresses = self.stresses
        disps = self.disp_field
        forces = self.force_field


        x = coords[:, 0]
        y = coords[:, 1]

        if shape_outline is None:
            shape_outline = np.append(self.mesh.all_corners, self.mesh.all_corners[0, :][None, :], axis=0)


        data = strains
        labels = ["$\epsilon_{11}$", "$\epsilon_{22}$", "$\gamma_{12}$", "$|\epsilon|$"]
        plotting_3(x, y, data, labels, shape_outline)

        data = stresses
        labels = ["$\sigma_{11}$", "$\sigma_{22}$", "$\sigma_{12}$", "$|\sigma|$"]
        plotting_3(x, y, data, labels, shape_outline)

        data = disps
        labels = ["$x_1$ displacement", "$x_2$ displacement", "Total displacement"]
        plotting_2(x, y, data, labels, shape_outline)

        data = forces
        labels = ["$F_{x_1}$", "$F_{x_2}$", "$|F|$"]
        plotting_2(x, y, data, labels, shape_outline)


