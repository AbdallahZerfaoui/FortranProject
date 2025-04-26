Okay, here is a detailed breakdown of the proposed C++ class structure using Markdown, outlining inheritance, responsibilities, attributes, methods, and key relationships.

---

# C++ Class Structure for Finite Difference Solver

This document outlines a proposed C++ class structure for translating the provided Fortran code, incorporating object-oriented principles, support for sequential/parallel execution, and steady/unsteady problems using inheritance and composition.

## Design Approach

The core idea is to define abstract base classes (interfaces) for fundamental concepts like Vectors, Matrices, and Solvers. Specific concrete classes will then implement these interfaces for sequential or parallel execution. A top-level `Simulation` class will orchestrate the process, choosing the appropriate concrete implementations based on configuration.

## Class Breakdown

### 1. Abstract Base Classes (Interfaces)

These classes define the contract for essential mathematical objects and operations. They declare pure virtual methods.

#### **`VectorBase`**

*   **Responsibility:** Defines the interface for vector operations.
*   **Inheritance:** None (Base Class).
*   **Attributes:**
    *   None (Pure interface).
*   **Methods:**
    *   `virtual ~VectorBase() = default;`
        *   *Description:* Virtual destructor for proper cleanup of derived classes.
    *   `virtual size_t getGlobalSize() const = 0;`
        *   *Description:* Returns the total number of elements in the global vector.
    *   `virtual size_t getLocalSize() const = 0;`
        *   *Description:* Returns the number of elements stored locally on the current process (for parallel). For sequential, this is the same as global size.
    *   `virtual double* getLocalData() = 0;`
        *   *Description:* Provides write access to the raw local data buffer.
    *   `virtual const double* getLocalData() const = 0;`
        *   *Description:* Provides read access to the raw local data buffer.
    *   `virtual double dot(const VectorBase& other) const = 0;`
        *   *Description:* Computes the global dot product (`this . other`). This requires communication in parallel implementations.
    *   `virtual void axpy(double alpha, const VectorBase& y) = 0;`
        *   *Description:* Performs the operation `this = this + alpha * y`. Operates on local data.
    *   `virtual void scale(double alpha) = 0;`
        *   *Description:* Performs the operation `this = alpha * this`. Operates on local data.
    *   `virtual void copy(const VectorBase& other) = 0;`
        *   *Description:* Copies the contents of `other` to `this`. Operates on local data.
    *   `virtual double norm() const = 0;`
        *   *Description:* Computes the global L2 norm (`sqrt(this . this)`). Requires communication in parallel.
    *   *(Other potential methods like element-wise operations: add, subtract, etc.)*
*   **Relationships:**
    *   Inherited by: `SequentialVector`, `DistributedVector`.
    *   Used by: `MatrixBase::multiply`, `SolverBase::solve`, `ProblemAssembler`, `Simulation`.

#### **`MatrixBase`**

*   **Responsibility:** Defines the interface for matrix operations, primarily matrix-vector multiplication.
*   **Inheritance:** None (Base Class).
*   **Attributes:**
    *   None (Pure interface).
*   **Methods:**
    *   `virtual ~MatrixBase() = default;`
        *   *Description:* Virtual destructor.
    *   `virtual void multiply(const VectorBase& input, VectorBase& output) const = 0;`
        *   *Description:* Performs the matrix-vector product: `output = this * input`. Implementations must handle their specific storage and potentially parallel communication needs (e.g., assuming input vector has halos).
*   **Relationships:**
    *   Inherited by: `SequentialSparseMatrix`, `ParallelSparseMatrix`.
    *   Used by: `SolverBase::solve`, `Simulation`.

#### **`SolverBase`**

*   **Responsibility:** Defines the interface for solving a linear system `A * U = F`.
*   **Inheritance:** None (Base Class).
*   **Attributes:**
    *   None (Pure interface).
*   **Methods:**
    *   `virtual ~SolverBase() = default;`
        *   *Description:* Virtual destructor.
    *   `virtual int solve(const MatrixBase& A, VectorBase& U, const VectorBase& F) = 0;`
        *   *Description:* Solves the linear system. `A` is the matrix, `F` is the right-hand side, and `U` is the input (initial guess) and output (solution). Returns the number of iterations or a status code. Implementations will cast base class references to concrete types.
*   **Relationships:**
    *   Inherited by: `SequentialCG`, `ParallelCG`.
    *   Used by: `Simulation`.

### 2. Concrete Classes: Sequential Implementation

These classes provide the specific implementation for a single-process, sequential execution.

#### **`SequentialVector : public VectorBase`**

*   **Responsibility:** Implements `VectorBase` using a standard `std::vector`. Stores the entire global vector.
*   **Inheritance:** `public VectorBase`.
*   **Attributes:**
    *   `std::vector<double> data;`
        *   *Description:* Stores all elements of the vector.
*   **Methods:**
    *   `SequentialVector(size_t global_size);`
        *   *Description:* Constructor, allocates the internal `std::vector`.
    *   Implementations for all `VectorBase` pure virtual methods using `data.size()`, `data.data()`, `std::inner_product`, and direct vector operations.
*   **Relationships:**
    *   Instantiated by: `Simulation`.
    *   Used by: `SequentialSparseMatrix`, `SequentialCG`, `ProblemAssembler`.

#### **`SequentialSparseMatrix : public MatrixBase`**

*   **Responsibility:** Implements `MatrixBase` for the 5-point stencil matrix, storing the full matrix diagonals for sequential access.
*   **Inheritance:** `public MatrixBase`.
*   **Attributes:**
    *   `std::vector<std::array<double, 5>> data;`
        *   *Description:* Stores the 5 diagonal coefficients for each row of the *entire* global matrix. Size is global `n`. Array indices (0-4) correspond to stencil points (i-Nx, i-1, i, i+1, i+Nx).
    *   `int Nx;`
        *   *Description:* Needed to calculate index offsets (`i-Nx`, `i+Nx`) in the `multiply` method.
*   **Methods:**
    *   `SequentialSparseMatrix(const GridParameters& grid);`
        *   *Description:* Constructor, fills the `data` vector based on the grid and diffusion coefficient.
    *   `void multiply(const VectorBase& input, VectorBase& output) const override;`
        *   *Description:* Implements matrix-vector product (Fortran `prodMV`). Casts input/output `VectorBase&` to `const SequentialVector&`/`SequentialVector&`. Operates on the full vectors.
*   **Relationships:**
    *   Instantiated by: `Simulation`.
    *   Used by: `SequentialCG`.

#### **`SequentialCG : public SolverBase`**

*   **Responsibility:** Implements `SolverBase` for the Conjugate Gradient algorithm using sequential vector and matrix types.
*   **Inheritance:** `public SolverBase`.
*   **Attributes:**
    *   `double epsilon;`
        *   *Description:* Convergence tolerance for the residual norm.
*   **Methods:**
    *   `SequentialCG(double tolerance);`
        *   *Description:* Constructor.
    *   `int solve(const MatrixBase& A, VectorBase& U, const VectorBase& F) override;`
        *   *Description:* Implements the CG algorithm. Requires casting input `MatrixBase&`, `VectorBase&` references to `SequentialSparseMatrix&`, `SequentialVector&` to perform operations. Uses sequential vector methods (`dot`, `axpy`, etc.) and `SequentialSparseMatrix::multiply`.
*   **Relationships:**
    *   Instantiated by: `Simulation`.

### 3. Concrete Classes: Parallel Implementation

These classes provide the specific implementation for distributed-memory, parallel execution using MPI.

#### **`MpiHandler`**

*   **Responsibility:** Manages the MPI environment (initialization, finalization) and provides core MPI communication wrappers and process information.
*   **Inheritance:** None.
*   **Attributes:**
    *   `int rank;`
        *   *Description:* The rank of the current process in `MPI_COMM_WORLD`.
    *   `int size;`
        *   *Description:* The total number of processes in `MPI_COMM_WORLD`.
    *   `MPI_Comm world_comm;`
        *   *Description:* The global MPI communicator.
*   **Methods:**
    *   `MpiHandler(int argc, char** argv);`
        *   *Description:* Constructor. Calls `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`.
    *   `~MpiHandler();`
        *   *Description:* Destructor. Calls `MPI_Finalize`.
    *   `int getRank() const;`
    *   `int getSize() const;`
    *   `void calculateLocalRange(int global_size, int& local_start_idx, int& local_end_idx) const;`
        *   *Description:* Implements the domain decomposition logic (`charge`) to determine the 0-based global index range for the local portion on this rank.
    *   `double globalAllReduceSum(double local_value) const;`
        *   *Description:* Performs `MPI_Allreduce(MPI_SUM)` on a double.
    *   `void sendRecv(const double* send_data, int send_count, int dest, int send_tag, double* recv_data, int recv_count, int source, int recv_tag) const;`
        *   *Description:* Wrapper for `MPI_Sendrecv`, used for halo exchange.
    *   `void barrier() const;`
        *   *Description:* Wrapper for `MPI_Barrier`.
    *   `void gatherV(const double* local_data, int local_count, double* global_data, const int* recvcounts, const int* displs, int root_rank) const;`
        *   *Description:* Wrapper for `MPI_Gatherv` used to collect distributed vector data onto a root process.
    *   *(Other MPI wrappers as needed)*
*   **Relationships:**
    *   Instantiated by: `Simulation` (conditionally).
    *   Used by: `DistributedVector`, `ParallelCG`, `ProblemAssembler` (potentially), `Simulation`.

#### **`DistributedVector : public VectorBase`**

*   **Responsibility:** Implements `VectorBase` for distributed vectors, managing local data and halo regions for parallel stencil operations.
*   **Inheritance:** `public VectorBase`.
*   **Attributes:**
    *   `std::vector<double> local_data;`
        *   *Description:* Stores the elements of the vector that belong to this process's domain.
    *   `std::vector<double> global_data_with_halos;`
        *   *Description:* A buffer used for matrix-vector products. It contains the `local_data` plus `Nx` ghost cell values from neighboring processes on both sides.
    *   `size_t global_n;`
        *   *Description:* The total number of elements across all processes.
    *   `int local_start_idx;`
        *   *Description:* The 0-based global index of the first element in `local_data`.
    *   `int local_end_idx;`
        *   *Description:* The 0-based global index of the last element in `local_data`.
    *   `int Nx;`
        *   *Description:* Needed to determine the size of the halo regions (Nx elements on left and right).
    *   `const MpiHandler* mpi_;`
        *   *Description:* Pointer to the MPI handler instance for communication.
*   **Methods:**
    *   `DistributedVector();`
        *   *Description:* Default constructor.
    *   `void allocate(size_t global_size, const GridParameters& grid, const MpiHandler& mpi);`
        *   *Description:* Initializes internal sizes, calculates local range using `mpi.calculateLocalRange`, and allocates `local_data` and `global_data_with_halos`. Stores `mpi` pointer.
    *   `void updateHalos() const;`
        *   *Description:* Performs MPI communication (`mpi_->sendRecv`) to exchange boundary data with neighbors and fill the halo regions in `global_data_with_halos`. This MUST be called before a parallel matrix-vector product.
    *   `const double* getGlobalDataWithHalos() const;`
        *   *Description:* Provides read access to the `global_data_with_halos` buffer.
    *   `size_t getHaloOffset() const;`
        *   *Description:* Returns the offset within `global_data_with_halos` where the local data begins (typically `Nx`).
    *   Implementations for `VectorBase` pure virtual methods:
        *   `getGlobalSize`, `getLocalSize`, `getLocalData`: Use `global_n`, `local_data.size()`, `local_data.data()`.
        *   `dot`: Calculates local dot product contribution, then uses `mpi_->globalAllReduceSum`.
        *   `axpy`, `scale`, `copy`: Operate *only* on `local_data`.
        *   `norm`: Calculates local norm contribution, then uses `mpi_->globalAllReduceSum` and `sqrt`.
*   **Relationships:**
    *   Instantiated by: `Simulation` (conditionally).
    *   Used by: `ParallelSparseMatrix`, `ParallelCG`, `ProblemAssembler`.
    *   Uses: `MpiHandler`.

#### **`ParallelSparseMatrix : public MatrixBase`**

*   **Responsibility:** Implements `MatrixBase` for the 5-point stencil matrix, storing only the local block of diagonals for parallel access.
*   **Inheritance:** `public MatrixBase`.
*   **Attributes:**
    *   `std::vector<std::array<double, 5>> local_data;`
        *   *Description:* Stores the 5 diagonal coefficients for the rows owned by this process. Size is local `local_size`.
    *   `int local_start_idx, local_end_idx;`
        *   *Description:* Global index range covered by this matrix block.
    *   `int Nx;`
        *   *Description:* Needed for index offsets during multiplication.
    *   `int global_n;`
        *   *Description:* Total global size (needed for boundary condition logic during fill).
*   **Methods:**
    *   `ParallelSparseMatrix(const GridParameters& grid, int local_start, int local_end);`
        *   *Description:* Constructor. Fills the `local_data` vector based on the grid, diffusion coefficient, and the local index range.
    *   `void multiply(const VectorBase& input, VectorBase& output) const override;`
        *   *Description:* Implements matrix-vector product (Fortran `prodMV_para`). Casts input/output `VectorBase&` to `const DistributedVector&`/`DistributedVector&`. **Assumes `input`'s halos (`global_data_with_halos`) are already updated.** Uses `local_data` and `input.getGlobalDataWithHalos()` to compute the product and writes to `output.getLocalData()`.
*   **Relationships:**
    *   Instantiated by: `Simulation` (conditionally).
    *   Used by: `ParallelCG`.
    *   Operates on: `DistributedVector`.

#### **`ParallelCG : public SolverBase`**

*   **Responsibility:** Implements `SolverBase` for the Conjugate Gradient algorithm using distributed vector and parallel matrix types, incorporating necessary parallel communication.
*   **Inheritance:** `public SolverBase`.
*   **Attributes:**
    *   `double epsilon;`
        *   *Description:* Convergence tolerance.
    *   `const MpiHandler* mpi_;`
        *   *Description:* Pointer to the MPI handler for communication (needed for global checks, potentially).
*   **Methods:**
    *   `ParallelCG(double tolerance, const MpiHandler& mpi);`
        *   *Description:* Constructor, stores `mpi` pointer.
    *   `int solve(const MatrixBase& A, VectorBase& U, const VectorBase& F) override;`
        *   *Description:* Implements the CG algorithm. Requires casting to `ParallelSparseMatrix&`, `DistributedVector&`. Key differences from sequential:
            *   **Calls `input_vector.updateHalos()` before `A.multiply(input_vector, output_vector)`.**
            *   Uses `DistributedVector::dot()` and `DistributedVector::norm()` which perform global reductions.
            *   Uses `DistributedVector::axpy()`, `scale()`, `copy()` which operate on local data.
*   **Relationships:**
    *   Instantiated by: `Simulation` (conditionally).
    *   Uses: `DistributedVector`, `ParallelSparseMatrix`, `MpiHandler`.

### 4. Utility Classes

These classes provide supporting data structures and functions that are not part of the core linear algebra hierarchy but are essential for setting up and defining the problem.

#### **`GridParameters`**

*   **Responsibility:** Stores grid dimensions, physical domain size, diffusion coefficient, and derived quantities (`dx`, `dy`, `n`). Provides index mapping utilities.
*   **Inheritance:** None.
*   **Attributes:**
    *   `int Nx, Ny;`
    *   `double Lx, Ly;`
    *   `double D;`
    *   `double dx, dy;`
    *   `int n;` // Total number of grid points (Nx * Ny)
*   **Methods:**
    *   `GridParameters(int nx, int ny, double lx, double ly, double d);`
        *   *Description:* Constructor. Calculates `dx`, `dy`, `n`.
    *   `void globalIndexToGrid(int k, int& i, int& j) const;`
        *   *Description:* Maps 1-based global index `k` to 1-based grid coordinates `(i, j)` (like Fortran `passage`).
    *   `int gridIndexToGlobal(int i, int j) const;`
        *   *Description:* Maps 1-based grid coordinates `(i, j)` to 1-based global index `k`.
    *   *(Getters for all attributes)*
*   **Relationships:**
    *   Instantiated by: `Simulation`.
    *   Used by: Matrix/Vector constructors (`SequentialSparseMatrix`, `ParallelSparseMatrix`, `DistributedVector::allocate`), `ProblemAssembler`.

#### **`ProblemDefinition`**

*   **Responsibility:** Encapsulates the mathematical functions defining the source term and boundary conditions.
*   **Inheritance:** None.
*   **Attributes:**
    *   (May hold copies or references to `Lx`, `Ly` if functions depend on them, or just pass `GridParameters` to methods).
*   **Methods:**
    *   `double evalF_steady(double x, double y) const;`
        *   *Description:* Evaluates the steady source term `f1` at `(x, y)`.
    *   `double evalF_unsteady(double x, double y, double t) const;`
        *   *Description:* Evaluates the unsteady source term `f1_insta` at `(x, y, t)`.
    *   `double evalG(double x, double y) const;`
        *   *Description:* Evaluates boundary condition function `g`.
    *   `double evalH(double x, double y) const;`
        *   *Description:* Evaluates boundary condition function `h`.
    *   *(Note: These functions might need access to Lx/Ly, which could be passed as arguments or via a pointer/reference to GridParameters)*
*   **Relationships:**
    *   Instantiated by: `Simulation`.
    *   Used by: `ProblemAssembler`.

#### **`ProblemAssembler`**

*   **Responsibility:** Fills the right-hand side vector(s) based on the problem definition, boundary conditions, and potentially previous time step solution.
*   **Inheritance:** None.
*   **Attributes:**
    *   `const GridParameters* grid_;`
        *   *Description:* Pointer to grid parameters.
    *   `const ProblemDefinition* problem_;`
        *   *Description:* Pointer to problem functions.
*   **Methods:**
    *   `ProblemAssembler(const GridParameters& grid, const ProblemDefinition& problem);`
        *   *Description:* Constructor, stores pointers.
    *   `void fillRHS_steady(VectorBase& F, const MpiHandler* mpi);`
        *   *Description:* Fills the RHS vector `F` for the steady problem. Iterates over local indices (using `mpi` pointer if not null), calls `ProblemDefinition` functions, applies boundary conditions using `GridParameters`. Operates on the local part of `F`.
    *   `void fillRHS_unsteady(VectorBase& F, const VectorBase& U_prev, double dt, int k_time, const MpiHandler* mpi);`
        *   *Description:* Fills the RHS vector for the unsteady problem (`U_prev + dt * F_source`). Iterates over local indices, calculates `U_prev(local_index) + dt * problem_->evalF_unsteady(...)`, applies boundary conditions contributions. Operates on the local part of `F`.
*   **Relationships:**
    *   Instantiated by: `Simulation`.
    *   Used by: `Simulation`.
    *   Uses: `GridParameters`, `ProblemDefinition`.
    *   Operates on: `VectorBase`.

### 5. Orchestration Class

This is the top-level class that controls the simulation flow, reads configuration, creates objects, runs the solver(s), and handles I/O.

#### **`Simulation`**

*   **Responsibility:** Reads configuration, sets up the simulation environment (MPI, grid, problem), creates the appropriate matrix, vectors, and solver based on configuration, runs the time loop (if unsteady), calls the solver, and handles results saving.
*   **Inheritance:** None.
*   **Attributes:**
    *   `// Configuration Parameters (read from input)`
    *   `int Nx, Ny, nt;`
    *   `double Lx, Ly, D, dt, epsilon;`
    *   `bool is_parallel;`
    *   `bool is_unsteady;`
    *   `// Core Objects`
    *   `std::unique_ptr<MpiHandler> mpi_handler_;`
        *   *Description:* Manages MPI. Created only if `is_parallel` is true.
    *   `GridParameters grid_params_;`
    *   `ProblemDefinition problem_def_;`
    *   `ProblemAssembler assembler_;`
    *   `std::unique_ptr<MatrixBase> A_;`
        *   *Description:* Holds the matrix (either `SequentialSparseMatrix` or `ParallelSparseMatrix`).
    *   `std::unique_ptr<VectorBase> U_, F_;`
        *   *Description:* Solution vector (`U`) and RHS vector (`F`). Types are `SequentialVector` or `DistributedVector`.
    *   `std::unique_ptr<VectorBase> Uk_;`
        *   *Description:* Previous time step solution vector (used in unsteady, type matches `U_`).
    *   `std::unique_ptr<SolverBase> solver_;`
        *   *Description:* Holds the solver (either `SequentialCG` or `ParallelCG`).
*   **Methods:**
    *   `Simulation(int argc, char** argv, const Config& config);`
        *   *Description:* Constructor. Reads configuration, initializes `mpi_handler_` if parallel, creates `grid_params_`, `problem_def_`, `assembler_`.
    *   `void setup_problem();`
        *   *Description:* Allocates and initializes `A_`, `U_`, `F_`, `Uk_` (if unsteady), and `solver_` based on `is_parallel` and `is_unsteady` flags, using `unique_ptr`. Fills the matrix `A_`. Initializes `U_` and `Uk_` (e.g., to zero).
    *   `void run();`
        *   *Description:* Main driver method. Calls `run_steady()` or `run_unsteady()` based on `is_unsteady` flag.
    *   `void run_steady();`
        *   *Description:* Runs the steady simulation. Calls `assembler_.fillRHS_steady()`. Calls `solver_->solve(A_, U_, F_)`. Calls `save_results()`.
    *   `void run_unsteady();`
        *   *Description:* Runs the unsteady simulation. Implements the time loop `for k_time = 0 to nt-1`. Inside loop: calls `assembler_.fillRHS_unsteady()` to compute the RHS using `Uk_`. Defines the effective unsteady matrix-vector product operator (e.g., via a helper method or lambda). Calls `solver_->solve()` using this unsteady operator and the computed RHS. Updates `Uk_ = U_`. Saves results periodically.
    *   `void calculate_unsteady_multiply(const VectorBase& input, VectorBase& output, double dt, const MatrixBase& A) const;`
        *   *Description:* Helper method to compute `output = input + dt * (A * input)`. This is the matrix-vector product for the implicit unsteady system. Calls `A.multiply()` internally.
    *   `void save_results(const std::string& filename) const;`
        *   *Description:* Saves the final (or intermediate) solution vector `U_`. Handles gathering distributed data onto rank 0 using `mpi_handler_` if `is_parallel` is true, then writes to file (e.g., VTK or text).
    *   `~Simulation() = default;`
        *   *Description:* Destructor. `unique_ptr`s automatically handle deallocation. `MpiHandler` destructor handles `MPI_Finalize`.
*   **Relationships:**
    *   Manages instances of: `MpiHandler` (pointer), `GridParameters`, `ProblemDefinition`, `ProblemAssembler`.
    *   Manages polymorphic instances via `unique_ptr`: `A_` (`MatrixBase`), `U_`, `F_`, `Uk_` (`VectorBase`), `solver_` (`SolverBase`).

## Summary of Relationships

*   **Inheritance:**
    *   `SequentialVector`, `DistributedVector` inherit from `VectorBase`.
    *   `SequentialSparseMatrix`, `ParallelSparseMatrix` inherit from `MatrixBase`.
    *   `SequentialCG`, `ParallelCG` inherit from `SolverBase`.
*   **Composition/Usage:**
    *   `Simulation` manages `MpiHandler`, `GridParameters`, `ProblemDefinition`, `ProblemAssembler`, and polymorphic objects via `unique_ptr`s to the Base classes.
    *   Parallel classes (`DistributedVector`, `ParallelCG`) use a pointer/reference to `MpiHandler` for communication.
    *   Concrete Solver classes (`SequentialCG`, `ParallelCG`) operate on concrete Matrix and Vector types (obtained via casting from Base references/pointers).
    *   `ParallelSparseMatrix`'s `multiply` method is specifically designed to work with `DistributedVector`s, assuming halos are updated.
    *   `ProblemAssembler` uses `GridParameters` and `ProblemDefinition` to fill `VectorBase` objects.
    *   The `Simulation` class implements the main logic, including the time loop (if unsteady) and the logic for using the correct matrix-vector product operator (`A` vs `I+dt*A`) within the solver calls.

This detailed structure provides a robust framework for implementing the solver while clearly separating concerns and allowing for the different variations (sequential/parallel, steady/unsteady).
