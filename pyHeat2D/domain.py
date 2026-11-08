import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any, Callable


class Domain():
    def __init__(self, Lx:float, Ly:float):
        self.Lx = Lx
        self.Ly = Ly
        self.boundaries = {}  # Store indices for each boundary (west, east, north, south)
        self.dirichlet_indices = []  # Selected Dirichlet condition indices
        self.neumann_indices = []  # Selected Neumann condition indices
        self.dict_dirichlet_indices = {} # store dirichlet indices for each boundary
        self.dict_neumann_indices = {} # store neumann indices for each boundary

        self.dirichlet_coordinates = []  # Selected Dirichlet condition coordinates
        self.neumann_coordinates = []  # Selected Neumann condition coordinates
        self.dict_dirichlet_coordinates = {} # store dirichlet coordinates for each boundary
        self.dict_neumann_coordinates = {} # store neumann coordinates for each boundary

    def create_mesh(self, nx:int, ny:int):
        """
        :param nx:
        :param ny:
        :return: Xc, Yc, X, Y
        """
        xmin = -self.Lx / 2
        xmax = self.Lx / 2
        ymin = -self.Ly / 2
        ymax = self.Ly / 2
        dx = self.Lx/(nx-1)
        dy = self.Ly/(ny-1)
        ncx = nx-1
        ncy = ny-1
        xc = np.linspace(xmin + dx / 2, xmax - dx / 2, ncx)
        yc = np.linspace(ymin + dy / 2, ymax - dy / 2, ncy)
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        # Meshing the domain
        centroid_position = np.meshgrid(xc, yc, indexing='ij')
        vertices_position= np.meshgrid(x, y, indexing='ij')
        # Indexing each cell after the mesh
        centroid_number = np.arange((ncx * ncy))
        centroid_number = np.reshape(centroid_number, (ncx, ncy))

        X, Y = vertices_position

        self.boundaries = {
            "west": np.ravel_multi_index(np.argwhere(X == xmin).T, (nx, ny)),
            "east": np.ravel_multi_index(np.argwhere(X == xmax).T, (nx, ny)),
            "south": np.ravel_multi_index(np.argwhere(Y == ymin).T, (nx, ny)),
            "north": np.ravel_multi_index(np.argwhere(Y == ymax).T, (nx, ny)),
        }

        # Combine all boundary vertices
        contour_indices = np.concatenate(list(self.boundaries.values()))
        return centroid_position, vertices_position, centroid_number, contour_indices


    def visualize_meshgrid(self, centroid_position: Tuple[np.ndarray, np.ndarray],
                           vertices_position: Tuple[np.ndarray, np.ndarray],
                           title: str = "Meshgrid Visualization with Boundary Directions") -> Any:
        """
        Visualize a meshgrid with centroids, vertices, and boundary vertices highlighted.

        Parameters:
            centroid_position: Tuple of centroid coordinates.
            vertices_position: Tuple of vertex coordinates.
            title (str): Title of the plot. Default is "Meshgrid Visualization".
        """
        Xc, Yc = centroid_position
        X, Y = vertices_position

        # Boundary coordinates
        west_coords = (X.flatten()[self.boundaries["west"]], Y.flatten()[self.boundaries["west"]])
        east_coords = (X.flatten()[self.boundaries["east"]], Y.flatten()[self.boundaries["east"]])
        south_coords = (X.flatten()[self.boundaries["south"]], Y.flatten()[self.boundaries["south"]])
        north_coords = (X.flatten()[self.boundaries["north"]], Y.flatten()[self.boundaries["north"]])

        plt.figure(figsize=(8, 6))
        plt.scatter(X, Y, color='blue', s=10, label='Grid vertices')
        plt.scatter(Xc, Yc, color='red', s=10, label='Centroids')
        plt.plot(*west_coords, color='cyan', linewidth=5, alpha=0.5, label='West boundary')
        plt.plot(*east_coords, color='magenta', linewidth=5, alpha=0.5, label='East boundary')
        plt.plot(*south_coords, color='green', linewidth=5, alpha=0.5, label='South boundary')
        plt.plot(*north_coords, color='orange', linewidth=5, alpha=0.5, label='North boundary')
        plt.plot(X, Y, color='lightgray', linewidth=0.5)  # Horizontal lines
        plt.plot(X.T, Y.T, color='lightgray', linewidth=0.5)  # Vertical lines
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(title)
        plt.legend(loc="best")
        #plt.axis('equal')
        plt.show()

    def get_boundary_indices(self):
        return self.boundaries

    def set_boundary_conditions(self, side: str,
            dirichlet_mask: Callable[[float, float], bool],
            neumann_mask: Callable[[float, float], bool],
            vertices_position: Tuple[np.ndarray, np.ndarray]):
        """
        Set mixed boundary conditions (Dirichlet and Neumann) on a specific side based on masks.

        Parameters:
            side (str): One of 'west', 'east', 'south', 'north'.
            dirichlet_mask (Callable): Mask function to filter vertices for Dirichlet conditions.
            neumann_mask (Callable): Mask function to filter vertices for Neumann conditions.
            vertices_position: Tuple of meshgrid vertex positions (X, Y).
        """
        if side not in self.boundaries:
            raise ValueError(f"Invalid side '{side}'. Choose from 'west', 'east', 'south', 'north'.")

        X, Y = vertices_position
        indices = self.boundaries[side]

        # Apply masks to filter indices for Dirichlet and Neumann

        dirichlet_indices = [int(idx) for idx in indices if dirichlet_mask(X.flatten()[idx], Y.flatten()[idx])]
        neumann_indices = [int(idx) for idx in indices if neumann_mask(X.flatten()[idx], Y.flatten()[idx])]

        dirichlet_coordinates = (X.flatten()[dirichlet_indices], Y.flatten()[dirichlet_indices])
        neumann_coordinates = (X.flatten()[neumann_indices], Y.flatten()[neumann_indices])
        # Update global lists
        self.dirichlet_indices.extend(dirichlet_indices)
        self.neumann_indices.extend(neumann_indices)
        self.dict_dirichlet_indices[side] = dirichlet_indices
        self.dict_neumann_indices[side] = neumann_indices

        self.dirichlet_coordinates.extend(dirichlet_coordinates)
        self.neumann_coordinates.extend(neumann_coordinates)
        self.dict_dirichlet_coordinates[side] = dirichlet_coordinates
        self.dict_neumann_coordinates[side] = neumann_coordinates


    def visualize_boundary_conditions(self, vertices_position: Tuple[np.ndarray, np.ndarray]):
        """
        Visualize the boundary conditions with Dirichlet and Neumann highlighted.

        Parameters:
            vertices_position: Tuple of vertex coordinates.
        """
        X, Y = vertices_position

        # Get coordinates for Dirichlet and Neumann conditions
        dirichlet_coords = (X.flatten()[self.dirichlet_indices], Y.flatten()[self.dirichlet_indices])
        neumann_coords = (X.flatten()[self.neumann_indices], Y.flatten()[self.neumann_indices])

        plt.figure(figsize=(8, 6))
        #plt.scatter(X, Y, color='blue', s=10, label='Grid vertices')
        plt.scatter(*dirichlet_coords, color='red', s=10, marker='s', label='Dirichlet condition')
        plt.scatter(*neumann_coords, color='blue', s=10, marker='s', label='Neumann condition')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Boundary Conditions Visualization")
        plt.legend(loc="best")
        #plt.axis('equal')
        plt.show()