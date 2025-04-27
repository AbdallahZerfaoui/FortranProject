from imports import *


def main():
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    print(config)

    # Initialize grid
    grid = GridParameters(config)
    print(f"Total number of grid points: {grid.n}")
    print(f"GridParameters spacing in x direction: {grid.dx}")
    print(f"GridParameters spacing in y direction: {grid.dy}")
    # Convert global index to grid index
    i, j = grid.global_index_to_grid(5)
    print(f"Global index 5 corresponds to grid index: ({i}, {j})")
    # Convert grid index to global index
    k = grid.grid_index_to_global(i, j)
    print(f"GridParameters index ({i}, {j}) corresponds to global index: {k}")
    
    # Test Sequential Vector
    u = SequentialVector(10)
    v = SequentialVector(10)
    print(f"Vector: {u.data}")
    print(f"dot product: {u.dot(v)}")
    u.axpy(2.0, v)
    print(f"After axpy: {u.data}")
    u.scale(0.5)
    print(f"After scale: {u.data}")
    print(f"Norm: {u.norm()}")
    
    


if __name__ == "__main__":
    main()
