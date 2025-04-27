from imports import *


def main():
    # Load configuration
	config_loader = ConfigLoader()
	config = config_loader.load_config()
	print(config)
	
	# Initialize grid
	grid = Grid(config)
	print(f"Total number of grid points: {grid.n}")
	print(f"Grid spacing in x direction: {grid.dx}")
	print(f"Grid spacing in y direction: {grid.dy}")
	# Convert global index to grid index
	i, j = grid.global_index_to_grid(5)
	print(f"Global index 5 corresponds to grid index: ({i}, {j})")
	# Convert grid index to global index
	k = grid.grid_index_to_global(i, j)
	print(f"Grid index ({i}, {j}) corresponds to global index: {k}")
 

if __name__ == "__main__":
    main()