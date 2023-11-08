import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from merlin.splint import CartesianGrid

cart_grid = CartesianGrid(grid_vectors=[[1.2,2.3,3.4],[5.4,1.9,6.5,8.9,7.1],[4.2,3.6,9.6,3.5]])
print(cart_grid)
