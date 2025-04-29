from src.imports import *


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
    i, j = grid.global_index_to_grid(27)
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
    
    # Test Sequential Sparse Matrix
    A = SequentialSparseMatrix(grid)
    A.populate()
    # print(f"Matrix : {A.matrix}")
    # u = SequentialVector(grid.n)
    # u.data = np.random.rand(grid.n)
    # f = A.multiply(u)
    # print(f"Matrix-vector product: {f.data}")
    
    # Test Vector filling
    F = SequentialVector(grid.n)
    problem = ProblemDefinition(grid, case="steady_polynomial")
    assembler = ProblemAssembler(grid, problem)
    assembler.fill_rhs_steady(F)
    # print(f"Filled vector F: {F.data}")

    # Tester Solver
    solver = SequentialSolver(tolerance=1e-9)
    U = SequentialVector(grid.n)
    U.data = np.zeros(grid.n)
    solver.solve(A, U, F)
    # print(f"Solved vector U: {U.data}")

    # Mesure solver speed
    # time_taken = solver.mesure_solver_speed(A, U, F)
    # print(f"Time taken to solve the system: {time_taken:.6f} seconds")
    
    #plot U
    # plt.imshow(U.data.reshape(grid.config["Nx"], grid.config["Ny"]), cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title("Solution U")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    
    

if __name__ == "__main__":
    my_files = [
        "main.py",
        "config.py",
        "grid.py",
        "vectors.py",
        "matrices.py",
        "solvers.py",
        "problem_definition.py",
        "assembly.py",
        __file__
    ]
    # my_files_regex = [f.replace('.', r'\.') for f in my_files]

    cProfile.run('main()', 'profile_stats')
    stats = pstats.Stats('profile_stats')
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    pattern = '|'.join(my_files)
    stats.print_stats(pattern)
        
