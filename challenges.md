Okay, let's break down the implementation into a series of smaller, manageable challenges. This follows a logical progression from the simplest version (sequential, steady) to the most complex (parallel, unsteady), layering in the parallel and unsteady features incrementally.

For each challenge, we'll mention the main tasks, the classes involved, and what the testing goal is for that stage.

---

### Implementation Challenges Breakdown

**Phase 1: Sequential Steady Solver Core**

This is the simplest version. Focus on getting the core numerical method (CG) and problem setup working without any parallelism or time dependence.

*   **Challenge 1.1: Basic Structure and Grid Parameters**
    *   **Task:** Create the main `Simulation` class skeleton. Implement the `GridParameters` class and the basic `ProblemDefinition` functions (`f1`, `g`, `h`). Implement the `passage` (index mapping) logic within `GridParameters`. Add a simple way to read `data` file parameters.
    *   **Classes:** `GridParameters`, `ProblemDefinition`, `Simulation`.
    *   **Test Goal:** Can successfully read parameters, calculate `dx`, `dy`, `n`, and map between 1D and 2D indices.

*   **Challenge 1.2: Sequential Vector Implementation**
    *   **Task:** Implement `SequentialVector` inheriting from `VectorBase`. Use `std::vector<double>` internally. Implement the required `VectorBase` methods (`getGlobalSize`, `getLocalSize`, `getLocalData`, `dot`, `axpy`, `scale`, `copy`, `norm`).
    *   **Classes:** `VectorBase`, `SequentialVector`.
    *   **Test Goal:** Can create sequential vectors, perform basic vector arithmetic (`+`, `-`, `* scalar`, `dot product`) correctly. Compare dot products/norms against known values or simple test vectors.

*   **Challenge 1.3: Sequential Matrix and Matrix-Vector Product**
    *   **Task:** Implement `SequentialSparseMatrix` inheriting from `MatrixBase`. Store the full matrix diagonals (`std::vector<std::array<double, 5>>`). Implement the constructor to fill the matrix coefficients based on `GridParameters` for the *steady* problem `A`. Implement the `multiply` method (`prodMV` logic).
    *   **Classes:** `MatrixBase`, `SequentialSparseMatrix`.
    *   **Test Goal:** Can create the sequential matrix. Can perform matrix-vector products correctly for known simple vectors or by hand calculation for small `Nx`, `Ny`.

*   **Challenge 1.4: Sequential Problem Assembly (Steady)**
    *   **Task:** Implement `ProblemAssembler`. Implement the `fillRHS_steady` method to calculate and fill the `SequentialVector` `F` based on `GridParameters`, `ProblemDefinition` functions, and boundary conditions.
    *   **Classes:** `ProblemDefinition`, `ProblemAssembler`, `SequentialVector`.
    *   **Test Goal:** Can correctly assemble the RHS vector `F` for the steady problem. Verify values at interior and boundary points.

*   **Challenge 1.5: Sequential Conjugate Gradient Solver (Steady)**
    *   **Task:** Implement `SequentialCG` inheriting from `SolverBase`. Implement the `solve` method following the CG algorithm. This method will use the `SequentialVector` and `SequentialSparseMatrix` methods internally (may require casting `VectorBase`/`MatrixBase` pointers/references).
    *   **Classes:** `SolverBase`, `SequentialCG`, `SequentialVector`, `SequentialSparseMatrix`.
    *   **Test Goal:** Can solve a simple known linear system (maybe not the full PDE yet). Can solve the steady PDE problem on a small grid and verify the solution against known analytic solutions or numerical solvers like NumPy/SciPy. Check convergence.

*   **Challenge 1.6: Sequential Steady Simulation Orchestration and I/O**
    *   **Task:** Implement the `Simulation::run_steady()` method. Instantiate `GridParameters`, `ProblemDefinition`, `ProblemAssembler`, `SequentialSparseMatrix`, `SequentialVector`s (`U`, `F`), and `SequentialCG`. Call assembler to fill `F`. Call solver to find `U`. Implement basic saving of the final `U` vector to a file (e.g., simple text format).
    *   **Classes:** `Simulation`, all sequential classes.
    *   **Test Goal:** Can run the full sequential steady simulation from parameter reading to saving the final solution. Verify the output file content.

---

**Phase 2: Parallel Infrastructure (MPI and Distributed Vectors)**

Introduce the MPI layer and the concept of distributed vectors with halos.

*   **Challenge 2.1: MPI Handler**
    *   **Task:** Implement `MpiHandler`. Initialize MPI in the constructor, finalize in the destructor. Implement `getRank()`, `getSize()`. Implement `calculateLocalRange` (the `charge` logic) for 0-based indexing. Add wrappers for essential MPI calls like `MPI_Allreduce`.
    *   **Classes:** `MpiHandler`.
    *   **Test Goal:** Can compile and run a simple MPI program that prints rank and size. Can verify `calculateLocalRange` distributes indices correctly across processes. Can perform a simple global sum reduction using `globalAllReduceSum`.

*   **Challenge 2.2: Distributed Vector Allocation and Local Access**
    *   **Task:** Implement `DistributedVector` inheriting from `VectorBase`. Implement the `allocate` method using `MpiHandler::calculateLocalRange` to determine local size and allocate `local_data`. Allocate the `global_data_with_halos` buffer (size = `local_size + 2*Nx`). Implement `getGlobalSize`, `getLocalSize`, `getLocalData`, `getGlobalDataWithHalos`, `getHaloOffset`.
    *   **Classes:** `DistributedVector`, `MpiHandler`, `GridParameters`.
    *   **Test Goal:** Can create and allocate `DistributedVector`s on multiple processes. Verify local sizes and start/end indices are correct per rank. Verify memory is allocated for halos.

*   **Challenge 2.3: Parallel Vector Operations (Global Dot, Norm, Local Ops)**
    *   **Task:** Implement `DistributedVector::dot`. Calculate the local dot product and then use `MpiHandler::globalAllReduceSum` to get the global sum. Implement `DistributedVector::norm` using the global dot product. Implement `axpy`, `scale`, `copy` to operate *only* on `local_data`.
    *   **Classes:** `DistributedVector`, `MpiHandler`.
    *   **Test Goal:** Create test `DistributedVector`s. Verify local `axpy`, `scale`, `copy` work correctly. Verify `dot` and `norm` produce the same results as their sequential counterparts when run on the same total data distributed across processes.

*   **Challenge 2.4: Halo Exchange**
    *   **Task:** Implement `DistributedVector::updateHalos`. Use `MpiHandler::sendRecv` to exchange `Nx` elements with neighboring processes. This is a crucial parallel communication step. For rank 0 and Np-1, handle the boundary cases (no left/right neighbor respectively). Fill the `global_data_with_halos` buffer.
    *   **Classes:** `DistributedVector`, `MpiHandler`.
    *   **Test Goal:** Create a test `DistributedVector` with predictable values. Run `updateHalos`. Verify the halo regions in `global_data_with_halos` are correctly populated with data from adjacent processes.

---

**Phase 3: Parallel Matrix and Matrix-Vector Product**

Adapt the matrix representation and its core multiplication for the parallel context.

*   **Challenge 3.1: Parallel Matrix Allocation and Assembly**
    *   **Task:** Implement `ParallelSparseMatrix` inheriting from `MatrixBase`. Store only the *local* portion of the matrix diagonals (`std::vector<std::array<double, 5>> local_data`). Implement the constructor to fill these local coefficients based on the local index range (`local_start_idx`, `local_end_idx`) and `GridParameters` for the *steady* problem `A`.
    *   **Classes:** `MatrixBase`, `ParallelSparseMatrix`, `GridParameters`.
    *   **Test Goal:** Can create `ParallelSparseMatrix` instances on each rank. Verify the local matrix coefficients are correct for the assigned range of global indices.

*   **Challenge 3.2: Parallel Matrix-Vector Product**
    *   **Task:** Implement `ParallelSparseMatrix::multiply`. This method must take a `DistributedVector` as input (or cast a `VectorBase` to `DistributedVector`) and assume its halos (`global_data_with_halos`) are up-to-date. Perform the matrix-vector multiplication using the `local_data` and the `global_data_with_halos` buffer. Write the result into the `local_data` of the output `DistributedVector`.
    *   **Classes:** `ParallelSparseMatrix`, `DistributedVector`.
    *   **Test Goal:** Create a test `ParallelSparseMatrix` and a test `DistributedVector`. Manually update the input vector's halos. Call `multiply`. Verify the local output vector segment is correct.

---

**Phase 4: Parallel Steady Solver and Simulation**

Combine the parallel components to solve the steady problem in parallel.

*   **Challenge 4.1: Parallel Conjugate Gradient Solver (Steady)**
    *   **Task:** Implement `ParallelCG` inheriting from `SolverBase`. Implement the `solve` method. This follows the same CG algorithm as `SequentialCG`, but crucial steps now use the parallel features:
        *   Vector operations (`dot`, `axpy`, `scale`, `copy`) use the `DistributedVector` implementations (which handle global reductions for dot/norm).
        *   *Crucially*, *before* calling the matrix multiply (`A.multiply(...)`), call `input_vector.updateHalos()`.
    *   **Classes:** `SolverBase`, `ParallelCG`, `DistributedVector`, `ParallelSparseMatrix`, `MpiHandler`.
    *   **Test Goal:** Can solve a simple system using `ParallelCG`. Can solve the steady PDE problem in parallel. Compare the number of iterations and the final solution (`U`) to the sequential version.

*   **Challenge 4.2: Parallel Steady Problem Assembly (Steady)**
    *   **Task:** Modify `ProblemAssembler::fillRHS_steady` to work with `DistributedVector`. It should fill *only* the local part of the `F` vector (`F.getLocalData()`) corresponding to the process's assigned indices.
    *   **Classes:** `ProblemAssembler`, `DistributedVector`, `GridParameters`, `ProblemDefinition`.
    *   **Test Goal:** Verify the local segment of `F` is correctly filled on each process.

*   **Challenge 4.3: Parallel Steady Simulation Orchestration and I/O**
    *   **Task:** Implement the `Simulation::run_parallel_steady()` method. Instantiate `MpiHandler`, then based on rank/size and `GridParameters`, instantiate `DistributedVector`s (`U`, `F`), `ParallelSparseMatrix`, and `ParallelCG`. Call assembler to fill `F`. Call solver to find `U`. Implement parallel saving: each rank saves its local part, OR use `MpiHandler::gatherV` (or similar) to collect the full vector onto rank 0 for saving (as done in the Fortran `remplissage_V_para_stationnaire.f90`).
    *   **Classes:** `Simulation`, all parallel classes.
    *   **Test Goal:** Can run the full parallel steady simulation with multiple processes. Verify the combined output solution matches the sequential solution. Measure performance and observe scaling (optional but good).

---

**Phase 5: Unsteady Extension**

Add the time dependence.

*   **Challenge 5.1: Problem Definition for Unsteady**
    *   **Task:** Modify `ProblemDefinition` to include `evalF_insta` (or update `evalF` to take time).
    *   **Classes:** `ProblemDefinition`.
    *   **Test Goal:** Can evaluate the unsteady source term correctly at different times.

*   **Challenge 5.2: Unsteady Problem Assembly**
    *   **Task:** Implement `ProblemAssembler::fillRHS_unsteady`. This method takes the previous time step's solution vector (`U_prev`), the time step size (`dt`), and the current time step index (`k_time`). It calculates the RHS for the system `(I + dt*A) * U_new = U_prev + dt * F_source`, meaning it calculates `U_prev + dt * F_source` where `F_source` comes from `ProblemDefinition::evalF_insta`. It fills the *local* part of the `F` vector used by the solver with this value.
    *   **Classes:** `ProblemAssembler`, `DistributedVector` (or `VectorBase`), `GridParameters`, `ProblemDefinition`.
    *   **Test Goal:** Verify the RHS vector is correctly assembled for the first time step and subsequent steps, including the `U_prev` contribution and the time-dependent source.

*   **Challenge 5.3: Unsteady Linear System Operation**
    *   **Task:** The system being solved iteratively is `(I + dt*A) * v`. The CG solver needs a way to compute this matrix-vector product. Instead of creating a new matrix class, implement this operation: given a vector `v`, compute `v + dt * A*v`. This requires calling the existing `A.multiply(v)` (which handles halos in parallel) and then performing an `axpy` operation. This can be a method in `Simulation` or passed as a lambda/functor to the solver. Let's add a method to `Simulation` for simplicity initially: `calculate_unsteady_multiply(const VectorBase& input, VectorBase& output, double dt, const MatrixBase& A) const`.
    *   **Classes:** `Simulation`, `MatrixBase`, `VectorBase`.
    *   **Test Goal:** Given a matrix `A`, a vector `v`, and a time step `dt`, verify `calculate_unsteady_multiply` correctly computes `v + dt*A*v`. This requires testing both sequential and parallel versions.

*   **Challenge 5.4: Integrate Unsteady Logic into Solver**
    *   **Task:** Modify `SequentialCG::solve` and `ParallelCG::solve` (or create new methods like `solve_unsteady`) to use the unsteady matrix-vector product operation (`calculate_unsteady_multiply`) instead of the direct `A.multiply`.
    *   **Classes:** `SequentialCG`, `ParallelCG`, `Simulation` (or wherever `calculate_unsteady_multiply` lives).
    *   **Test Goal:** Can run the CG solver using the `(I + dt*A)` operator. Verify it converges (may need different tolerance or more iterations).

*   **Challenge 5.5: Unsteady Simulation Orchestration and Time Loop**
    *   **Task:** Implement `Simulation::run_unsteady()`. Instantiate `Uk` (`U_prev`) `VectorBase` (and initialize it, e.g., to zero as in Fortran `Uk=0`). Implement the time loop (`for k_time = 0 to nt-1`). Inside the loop:
        *   Call `ProblemAssembler::fillRHS_unsteady` to get the RHS vector for the current time step (using `Uk_` and `dt`).
        *   Call the solver (`solver_->solve(...)`) using the unsteady multiply operation and the current RHS.
        *   Update `Uk_` with the new solution `U_`.
        *   Save the solution periodically (e.g., every few time steps or at the end).
    *   **Classes:** `Simulation`, all other classes involved in the unsteady path.
    *   **Test Goal:** Can run the full unsteady simulation for a specified number of time steps. Verify the solution evolves over time. Compare results against sequential/parallel steady solutions if the unsteady problem should converge to a steady state. Save multiple time steps and potentially visualize.

---

**Phase 6: Refinement and Polish**

Improve robustness, usability, and performance.

*   **Challenge 6.1: Command Line Argument Parsing**
    *   **Task:** Use a library (like `getopt` or `Boost.Program_options`) or manual parsing to read simulation parameters (Nx, Ny, Lx, Ly, D, nt, dt, epsilon, output file, sequential/parallel flag, steady/unsteady flag, Np if parallel) from the command line instead of a simple `data` file.
    *   **Test Goal:** Can launch the executable with different arguments and confirm parameters are set correctly.

*   **Challenge 6.2: Error Handling**
    *   **Task:** Add checks for file opening failures, allocation errors, MPI errors (using `MPI_Errhandler_set` or checking return codes), invalid input parameters (e.g., negative Nx/Ny, Np > n). Use exceptions or return status codes.
    *   **Test Goal:** Running with invalid input results in informative error messages and graceful exit. MPI errors are caught or reported.

*   **Challenge 6.3: Output Format**
    *   **Task:** Implement saving the output `U` in a more standard scientific format, like VTK, which can be easily visualized. This will likely involve gathering the distributed data onto rank 0 and then writing the VTK file.
    *   **Classes:** `Simulation`, `MpiHandler` (for gathering).
    *   **Test Goal:** Output files can be opened and visualized in tools like Paraview.

*   **Challenge 6.4: Performance Optimization**
    *   **Task:** Use profiling tools (like gprof, Valgrind/Cachegrind, or MPI-specific tools like Vampir/Score-P) to identify bottlenecks. Optimize critical sections, particularly the matrix-vector product and halo exchange. Ensure correct compiler flags are used (`-O3`, etc.).
    *   **Test Goal:** Measure execution time on different numbers of processes and grid sizes. Aim for reasonable parallel scaling.

---

This phased approach allows you to build and test components incrementally. Start with the solid sequential foundation, then tackle the parallel complexities piece by piece, and finally add the time-dependent layer. Good luck!
