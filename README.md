# Heat Equation Solver

This project implements a solver for the 2D heat equation using the finite difference method and the Conjugate Gradient algorithm. It supports sequential and parallel execution, as well as steady-state and unsteady problems.

The project is structured using C++ Object-Oriented Programming principles, inspired by the original Fortran code but refactored for clarity and maintainability.

## Equations Solved

The code solves the heat equation in a 2D domain $[0, L_x] \times [0, L_y]$:

$\frac{\partial u}{\partial t} - D \Delta u = f(x, y, t)$

with boundary conditions:
$u|_{\Gamma_0} = g(x, y, t)$ (on boundaries with outer normal in the y-direction)
$u|_{\Gamma_1} = h(x, y, t)$ (on boundaries with outer normal in the x-direction)

For the steady-state problem ($\frac{\partial u}{\partial t} = 0$), the equation reduces to:
$- D \Delta u = f(x, y)$

## Project Structure

*   `include/`: Contains header files defining classes and interfaces.
    *   `GridGeometry.hpp`: Defines the grid structure and index mapping.
    *   `PhysicsParameters.hpp`: Defines physical constants like diffusion coefficient `D`.
    *   `EquationFunctions.hpp`: Provides source term and boundary condition functions.
    *   `SystemBase.hpp`: Base class for system assembly (matrix/RHS).
    *   `SequentialSystem.hpp`: Derives from `SystemBase` for sequential assembly.
    *   `ParallelSystem.hpp`: Manages local matrix/RHS and parallel matrix-vector products.
    *   `LinearOperatorBase.hpp`: Interface for matrix-vector operations used by CG (sequential).
    *   `MatrixOperator.hpp`: Implements `LinearOperatorBase` for a standard matrix A.
    *   `UnsteadyOperator.hpp`: Implements `LinearOperatorBase` for the `(I + dt*A)` operator.
    *   `SequentialCG.hpp`: Implements the sequential Conjugate Gradient solver.
    *   `ParallelCG.hpp`: Implements the parallel Conjugate Gradient solver using MPI.
*   `src/`: Contains the implementation files (.cpp) for classes and the main programs.
    *   `GridGeometry.cpp`
    *   `EquationFunctions.cpp`
    *   `SequentialSystem.cpp`
    *   `ParallelSystem.cpp`
    *   `SequentialCG.cpp`
    *   `ParallelCG.cpp`
    *   `SequentialHeatSolver.cpp`: Main program logic for the sequential solver.
    *   `ParallelHeatSolver.cpp`: Main program logic for the parallel solver (manages MPI).
*   `data`: Input file containing simulation parameters (Nx, Ny, Lx, Ly, D).
*   `Makefile`: Build instructions.

## Requirements

*   A C++11 compatible compiler (e.g., g++, clang++).
*   An MPI implementation (e.g., OpenMPI, MPICH) and its C++ bindings (`mpicxx`).
*   The Eigen library (version 3.3 or later recommended).

## Building the Project

1.  Make sure you have the Eigen library installed. You might need to adjust the `EIGEN_INCLUDE` variable in the `Makefile` to point to your Eigen headers (e.g., `-I/usr/local/include/eigen3`).
2.  Open a terminal in the project's root directory.
3.  Run `make` to build both the sequential and parallel executables.
    ```bash
    make
    ```
4.  To clean up generated object files:
    ```bash
    make clean
    ```
5.  To clean up executables and output files as well:
    ```bash
    make fclean
    ```
6.  To perform a clean and rebuild:
    ```bash
    make re
    ```

## Running the Solvers

1.  Create a `data` file in the project root directory with the required parameters:
    ```
    Nx Ny
    Lx Ly
    D
    ```
    Example `data` file:
    ```
    10 10
    1.0 1.0
    0.1
    ```

2.  Run the sequential steady solver (default in `main_sequential.cpp` skeleton):
    ```bash
    make run_sequential
    ```

3.  Run the parallel steady solver (default in `main_parallel.cpp` skeleton, using `NBR_PROC` processes):
    ```bash
    make run_parallel
    ```
    You can change the number of processes by modifying the `NBR_PROC` variable in the `Makefile` or by using `mpirun -np <number> ./heat_solver_parallel`.

4.  To run unsteady cases: The skeleton `main_sequential.cpp` and `main_parallel.cpp` have commented-out sections for running the unsteady solvers. Uncomment and configure them as needed, and potentially add command-line argument parsing to choose between steady/unsteady modes.

## Filling the Skeleton

The skeleton code provides empty method bodies (`{ // TODO: Implement ... }`) and comments (`// TODO: ...`) indicating where to translate the specific logic from your Fortran code into C++.

Start by implementing the simpler classes and methods first:

1.  `GridGeometry.cpp`: Implement the constructor and `indexToCoords`/`coordsToIndex`.
2.  `EquationFunctions.cpp`: Implement the static `default_f1_stationary`, `default_f1_unsteady`, `default_g`, and `default_h` functions.
3.  `SequentialSystem.cpp`: Implement `assembleMatrix`, `assembleRHS_steady`, and `assembleRHS_unsteady`.
4.  `MatrixOperator.cpp` and `UnsteadyOperator.cpp`: Implement the `multiply` methods using Eigen's operators.
5.  `SequentialCG.cpp`: Implement the Conjugate Gradient algorithm, using the `LinearOperatorBase::multiply` method for matrix-vector products and Eigen's built-in vector operations for dot products and norms.
6.  `SequentialHeatSolver.cpp`: Review `runSteady` and `runUnsteady` methods, ensuring they correctly call the assembler and solver, handle time stepping, and save results.
7.  `ParallelSystem.cpp`: Implement `determineLocalRange`, `assembleLocalMatrix`, `assembleLocalRHS_steady`, `assembleLocalRHS_unsteady`, and the crucial `parallelMatrixVectorProduct`. Implement parallel dot product and norm helpers.
8.  `ParallelCG.cpp`: Implement the parallel Conjugate Gradient algorithm, using the `ParallelSystem::parallelMatrixVectorProduct` and the MPI parallel dot product/norm helpers.
9.  `ParallelHeatSolver.cpp`: Review `runSteady` and `runUnsteady`, ensuring they handle parallel assembly, call the parallel CG, and manage parallel saving.

Remember to use `Eigen::SparseMatrix` for `m_A` and `m_A_local` for performance. Pay close attention to 0-based indexing in C++ vs. 1-based in Fortran, especially when translating loops and index mappings.
