import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from merlin.grid import CartesianGrid

# test grid
cart_grid = CartesianGrid(grid_vectors=[[1.2, 2.3, 3.4],[1.9, 5.4, 6.5, 7.1, 8.9],[3.5, 3.6, 4.2, 9.6]])
print(cart_grid)

grid_points = cart_grid.get_points()
print(grid_points)

