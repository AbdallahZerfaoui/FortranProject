Okay, here is a detailed skeleton of the C++ project based on the OOP structure we discussed. Each file includes comments explaining its purpose, the role of classes/methods, and where to integrate the logic from your Fortran code.

This structure assumes you will split your `my_code.txt` into individual `.f90` files first (e.g., `mod_pde_utils.f90`, `mod_cg_solver_sequential.f90`, etc.) as discussed, and then translate the logic from those specific files into the corresponding C++ classes and methods.

**Project File Structure:**

```
.
├── data                # Parameter file (Nx, Ny, Lx, Ly, D)
├── Makefile            # To build the project
├── src/                # Source files (.cpp)
│   ├── GridGeometry.cpp
│   ├── EquationFunctions.cpp
│   ├── SequentialSystem.cpp
│   ├── ParallelSystem.cpp
│   ├── SequentialCG.cpp
│   ├── ParallelCG.cpp
│   ├── SequentialHeatSolver.cpp   # Main program for sequential
│   └── ParallelHeatSolver.cpp     # Main program for parallel
├── include/            # Header files (.hpp)
│   ├── GridGeometry.hpp
│   ├── PhysicsParameters.hpp    # Header-only struct
│   ├── EquationFunctions.hpp
│   ├── SystemBase.hpp           # Base class for system assembly (optional, but good practice)
│   ├── SequentialSystem.hpp
│   ├── ParallelSystem.hpp
│   ├── LinearOperatorBase.hpp   # Base class/interface for matrix-vector product
│   ├── MatrixOperator.hpp       # Implements LinearOperator for A
│   ├── UnsteadyOperator.hpp     # Implements LinearOperator for I+dt*A
│   ├── ParallelLinearOperatorBase.hpp # Base for parallel mat-vec
│   ├── ParallelMatrixOperator.hpp # Implements ParallelLinearOperator for parallel A
│   ├── ParallelUnsteadyOperator.hpp # Implements ParallelLinearOperator for parallel I+dt*A
│   ├── CGSolverBase.hpp         # Base class for CG solver interface
│   ├── SequentialCG.hpp
│   └── ParallelCG.hpp
├── README.md           # Explanation of the project structure and how to build/run
```

**SKELETON FILES:**

---

**File: `include/PhysicsParameters.hpp`**

```cpp
#ifndef PHYSICS_PARAMETERS_HPP
#define PHYSICS_PARAMETERS_HPP

// Simple struct to hold physical parameters
struct PhysicsParameters {
    double D; // Diffusion coefficient

    // Constructor
    PhysicsParameters(double diffusion_coeff) : D(diffusion_coeff) {}

    // Add other physical parameters here if needed later (e.g., specific heat, density)
};

#endif // PHYSICS_PARAMETERS_HPP
```

---

**File: `include/GridGeometry.hpp`**

```cpp
#ifndef GRID_GEOMETRY_HPP
#define GRID_GEOMETRY_HPP

#include <iostream> // For printing/debugging (optional in header)

// Class to store grid dimensions and handle index mapping
class GridGeometry {
private:
    int m_Nx;       // Number of internal grid points in x-direction
    int m_Ny;       // Number of internal grid points in y-direction
    double m_Lx;    // Length of the domain in x-direction
    double m_Ly;    // Length of the domain in y-direction
    double m_dx;    // Grid spacing in x
    double m_dy;    // Grid spacing in y
    int m_n;        // Total number of internal grid points (Nx * Ny)

public:
    // Constructor
    GridGeometry(int Nx, int Ny, double Lx, double Ly);

    // Accessor methods
    int getNx() const { return m_Nx; }
    int getNy() const { return m_Ny; }
    double getLx() const { return m_Lx; }
    double getLy() const { return m_Ly; }
    double getDx() const { return m_dx; }
    double getDy() const { return m_dy; }
    int getNumNodes() const { return m_n; }

    // Method to map a 1D global index (0-based) to 2D grid coordinates (1-based)
    // Matches the logic of Fortran's 'passage' subroutine, but takes 0-based index.
    void indexToCoords(int k_0based, int& i_1based, int& j_1based) const;

    // Method to map 2D grid coordinates (1-based) to a 1D global index (0-based)
    // Useful for matrix/vector indexing.
    int coordsToIndex(int i_1based, int j_1based) const;

    // Add method to check if a 1D index is a boundary point if needed (derived from i,j)
};

#endif // GRID_GEOMETRY_HPP
```

---

**File: `src/GridGeometry.cpp`**

```cpp
#include "GridGeometry.hpp"
#include <cmath> // For pow (though simple multiplication is better for dx/dy)

// Constructor
GridGeometry::GridGeometry(int Nx, int Ny, double Lx, double Ly) :
    m_Nx(Nx), m_Ny(Ny), m_Lx(Lx), m_Ly(Ly)
{
    // TODO: Implement calculation of dx, dy, n based on Fortran logic
    // dx = Lx / (Nx + 1.0)
    // dy = Ly / (Ny + 1.0)
    // n = Nx * Ny
    m_dx = m_Lx / (static_cast<double>(m_Nx) + 1.0);
    m_dy = m_Ly / (static_cast<double>(m_Ny) + 1.0);
    m_n = m_Nx * m_Ny;

    std::cout << "Grid initialized: Nx=" << m_Nx << ", Ny=" << m_Ny
              << ", Lx=" << m_Lx << ", Ly=" << m_Ly
              << ", dx=" << m_dx << ", dy=" << m_dy
              << ", n=" << m_n << std::endl;
}

// Method to map a 1D global index (0-based) to 2D grid coordinates (1-based)
// Matches the logic of Fortran's 'passage' subroutine, but takes 0-based index.
void GridGeometry::indexToCoords(int k_0based, int& i_1based, int& j_1based) const {
    // k_0based is the 0-based index (0 to n-1)
    // Fortran's passage took l (1-based, 1 to n) and returned i, j (1-based).
    // i = 1 + int((l-1)/Nx)
    // j = 1 + mod(l-1, Nx)
    // With 0-based k, l-1 is just k.
    i_1based = 1 + k_0based / m_Nx; // row index (1-based)
    j_1based = 1 + k_0based % m_Nx; // column index (1-based)
}

// Method to map 2D grid coordinates (1-based) to a 1D global index (0-based)
// Useful for matrix/vector indexing.
int GridGeometry::coordsToIndex(int i_1based, int j_1based) const {
    // i_1based is 1 to Ny, j_1based is 1 to Nx
    // The 0-based index k corresponds to l-1 in Fortran.
    // l = (i-1)*Nx + j  (if i, j are 1-based)
    // k = l - 1 = (i-1)*Nx + j - 1
     return (i_1based - 1) * m_Nx + (j_1based - 1);
}

// Add implementation for other methods if declared in header
```

---

**File: `include/EquationFunctions.hpp`**

```cpp
#ifndef EQUATION_FUNCTIONS_HPP
#define EQUATION_FUNCTIONS_HPP

#include <functional> // For std::function
#include <cmath>      // For math functions (pow, cos, exp, etc.)

// Class to hold the source term and boundary condition functions
class EquationFunctions {
private:
    // Use std::function to allow setting these functions dynamically
    // Source term f(x, y, t)
    std::function<double(double x, double y, double t)> m_f_source;
    // Boundary condition g(x, y) on Gamma_0 (e.g., y=0, y=Ly)
    std::function<double(double x, double y)> m_g_boundary;
    // Boundary condition h(x, y) on Gamma_1 (e.g., x=0, x=Lx)
    std::function<double(double x, double y)> m_h_boundary;

public:
    // Constructor takes functions (or lambdas) for initialization
    // Example: EquationFunctions eq_funcs([](x,y,t){...}, [](x,y){...}, [](x,y){...});
    EquationFunctions(
        std::function<double(double x, double y, double t)> f_source_func,
        std::function<double(double x, double y)> g_func,
        std::function<double(double x, double y)> h_func
    ) : m_f_source(f_source_func), m_g_boundary(g_func), m_h_boundary(h_func) {}

    // Methods to get the value of the functions
    double getSource(double x, double y, double t) const { return m_f_source(x, y, t); }
    double getG(double x, double y) const { return m_g_boundary(x, y); }
    double getH(double x, double y) const { return m_h_boundary(x, y); }

    // --- Static "Default" Function Implementations ---
    // These static methods can be used to provide the functions
    // matching your Fortran code's implementations.

    // Matches f1 from mod_fonctions_instationnaire (stationary version)
    static double default_f1_stationary(double x, double y, double t);

    // Matches f1_insta from mod_fonctions_instationnaire (unsteady version)
    // Requires Lx, Ly, which should come from GridGeometry or passed somehow.
    // A better way is to capture Lx, Ly in the lambda when creating EquationFunctions.
    // For now, let's declare it with Lx, Ly params, assuming they are available.
    static double default_f1_unsteady(double x, double y, double t, double Lx, double Ly);

    // Matches g from mod_fonctions_instationnaire (hardcoded 0.0)
    static double default_g(double x, double y);

    // Matches h from mod_fonctions_instationnaire (hardcoded 1.0 or 0.0 - check which is desired)
    static double default_h(double x, double y);
};

#endif // EQUATION_FUNCTIONS_HPP
```

---

**File: `src/EquationFunctions.cpp`**

```cpp
#include "EquationFunctions.hpp"
#include <cmath> // Include cmath here as well for definitions

// Static "Default" Function Implementations

// Matches f1 from mod_fonctions_instationnaire (stationary version: 2*((x-x^2)+(y-y^2)))
double EquationFunctions::default_f1_stationary(double x, double y, double t) {
    // TODO: Implement the stationary f1 logic from Fortran
    // return 2.0 * ((x - std::pow(x, 2)) + (y - std::pow(y, 2)));
    return 2.0 * ((x - x*x) + (y - y*y)); // Avoid pow for simple squares
}

// Matches f1_insta from mod_fonctions_instationnaire (unsteady version)
// Assumes Lx and Ly are needed, passed here for clarity.
double EquationFunctions::default_f1_unsteady(double x, double y, double t, double Lx, double Ly) {
    // TODO: Implement the unsteady f1_insta logic from Fortran
    // f1_insta=exp(-1*(x-0.5*Lx)**2)*exp(-1*(y-0.5*Ly)**2)*cos(0.5*pi*t)
    const double PI = std::acos(-1.0);
    return std::exp(-1.0 * std::pow(x - 0.5 * Lx, 2)) *
           std::exp(-1.0 * std::pow(y - 0.5 * Ly, 2)) *
           std::cos(0.5 * PI * t);
}

// Matches g from mod_fonctions_instationnaire (hardcoded 0.0)
double EquationFunctions::default_g(double x, double y) {
    // TODO: Implement the g logic from Fortran (currently 0.0)
    return 0.0;
}

// Matches h from mod_fonctions_instationnaire (hardcoded 1.0 in one comment, 0.0 in another, let's pick 1.0 as it was active)
double EquationFunctions::default_h(double x, double y) {
    // TODO: Implement the h logic from Fortran (currently 1.0 in one active comment)
    return 1.0;
}
```

---

**File: `include/SystemBase.hpp` (Optional but good for polymorphism)**

```cpp
#ifndef SYSTEM_BASE_HPP
#define SYSTEM_BASE_HPP

#include <Eigen/Sparse> // Use Sparse matrix for efficiency

// Forward declarations
class GridGeometry;
class PhysicsParameters;
class EquationFunctions;

// Base class for assembling the linear system AU=F or (I+dt*A)U_new = U_old + dt*F
class SystemAssemblerBase {
protected:
    // Pointers/references to required input data (shared ownership is okay here)
    const GridGeometry& m_grid;
    const PhysicsParameters& m_physics;
    const EquationFunctions& m_eq_funcs;

public:
    // Constructor
    SystemAssemblerBase(const GridGeometry& grid, const PhysicsParameters& physics, const EquationFunctions& eq_funcs)
        : m_grid(grid), m_physics(physics), m_eq_funcs(eq_funcs) {}

    // Virtual destructor is important for base classes with derived classes
    virtual ~SystemAssemblerBase() = default;

    // Pure virtual method to assemble the matrix A
    // Derived classes must implement this (sequential vs parallel, sparse vs maybe dense)
    // Using SparseMatrix is recommended for performance.
    virtual void assembleMatrix(Eigen::SparseMatrix<double>& A_or_A_local) const = 0;

    // Pure virtual method to assemble the RHS vector F
    // Derived classes must implement this (sequential vs parallel, steady vs unsteady source)
    virtual void assembleRHS(Eigen::VectorXd& F_or_F_local, double time, const Eigen::VectorXd* U_old = nullptr) const = 0; // U_old needed for unsteady RHS

    // Optional: Add methods for matrix-vector product here if not using a separate LinearOperator class
};

#endif // SYSTEM_BASE_HPP
```

---

**File: `include/SequentialSystem.hpp`**

```cpp
#ifndef SEQUENTIAL_SYSTEM_HPP
#define SEQUENTIAL_SYSTEM_HPP

#include "SystemBase.hpp" // Inherit from base assembler
#include <Eigen/Sparse>

// Class to assemble the linear system for the sequential case
class SequentialSystemAssembler : public SystemAssemblerBase {
public:
    // Constructor
    SequentialSystemAssembler(const GridGeometry& grid, const PhysicsParameters& physics, const EquationFunctions& eq_funcs)
        : SystemAssemblerBase(grid, physics, eq_funcs) {}

    // Method to assemble the full matrix A (Eigen::SparseMatrix)
    // Implement logic from Fortran's matrix filling (program remplissage_V)
    void assembleMatrix(Eigen::SparseMatrix<double>& A) const override;

    // Method to assemble the full RHS vector F (Eigen::VectorXd)
    // Implement logic from Fortran's remplissage_F (for both steady and unsteady cases)
    // The 'time' parameter is used by unsteady source/BCs.
    // U_old is needed for the unsteady explicit RHS (U_old + dt*F).
    void assembleRHS(Eigen::VectorXd& F, double time, const Eigen::VectorXd* U_old = nullptr) const override;

    // Note: For the unsteady problem (I + dt*A) U_new = U_old + dt*F,
    // the matrix A is constant, but the RHS changes.
    // assembleMatrix() builds A.
    // assembleRHS() calculates U_old + dt*F where F includes boundary terms.
};

#endif // SEQUENTIAL_SYSTEM_HPP
```

---

**File: `src/SequentialSystem.cpp`**

```cpp
#include "SequentialSystem.hpp"
#include "GridGeometry.hpp"
#include "PhysicsParameters.hpp"
#include "EquationFunctions.hpp"
#include <cmath> // For pow
#include <iostream> // For debugging

// Method to assemble the full matrix A (Eigen::SparseMatrix)
// Implement logic from Fortran's matrix filling (program remplissage_V)
void SequentialSystemAssembler::assembleMatrix(Eigen::SparseMatrix<double>& A) const {
    // TODO: Implement matrix assembly using a 5-point stencil.
    // A should be an (n x n) sparse matrix.
    // Coefficients should match the scaled Fortran values:
    // Center: 2*D*(1/dx^2 + 1/dy^2)
    // Left/Right: -D/dx^2
    // Up/Down: -D/dy^2

    int n = m_grid.getNumNodes();
    double dx = m_grid.getDx();
    double dy = m_grid.getDy();
    double D = m_physics.D;
    int Nx = m_grid.getNx();
    int Ny = m_grid.getNy();

    A.resize(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 5)); // Reserve space for up to 5 non-zeros per row

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Loop through each node k (0-based index)
    for (int k = 0; k < n; ++k) {
        int i, j; // 1-based grid coordinates
        m_grid.indexToCoords(k, i, j); // Get 1-based i, j

        // Diagonal entry
        A.insert(k, k) = 2.0 * D * (1.0/dx2 + 1.0/dy2);

        // Off-diagonal entries (check for neighbors within the grid)

        // Neighbor to the left (k-1): exists if not on the left boundary (j > 1)
        if (j > 1) {
            A.insert(k, k - 1) = -D / dx2;
        }

        // Neighbor to the right (k+1): exists if not on the right boundary (j < Nx)
        if (j < Nx) {
             A.insert(k, k + 1) = -D / dx2;
        }

        // Neighbor above (k-Nx): exists if not on the top boundary (i > 1)
        if (i > 1) {
            A.insert(k, k - Nx) = -D / dy2;
        }

        // Neighbor below (k+Nx): exists if not on the bottom boundary (i < Ny)
        if (i < Ny) {
            A.insert(k, k + Nx) = -D / dy2;
        }
    }

    A.makeCompressed(); // Optimize storage

    std::cout << "Matrix A assembled (" << n << "x" << n << ")" << std::endl;
}

// Method to assemble the full RHS vector F (Eigen::VectorXd)
// Implement logic from Fortran's remplissage_F (for both steady and unsteady cases)
// The 'time' parameter is used by unsteady source/BCs.
// U_old is needed for the unsteady explicit RHS (U_old + dt*F).
void SequentialSystemAssembler::assembleRHS(Eigen::VectorXd& F, double time, const Eigen::VectorXd* U_old) const {
    // TODO: Implement RHS assembly.
    // If U_old is null, this is a steady case (RHS = f + BCs).
    // If U_old is not null, this is an unsteady step (RHS = U_old + dt * (f + BCs)).

    int n = m_grid.getNumNodes();
    double dx = m_grid.getDx();
    double dy = m_grid.getDy();
    double Lx = m_grid.getLx();
    double Ly = m_grid.getLy();
    int Nx = m_grid.getNx();
    int Ny = m_grid.getNy(); // Added Ny for bottom BC loop
    double dt = 0.0; // Time step - needs to be passed or stored if this is for unsteady
                     // Assuming for unsteady RHS calculation, the dt used for the time step is needed.
                     // Let's add dt as an argument or get it from the solver.
                     // For now, assume if U_old is not null, dt is needed. Let's add dt to signature.
                     // Redefining signature slightly for clarity: assembleRHS(F, time, dt_step, U_old).
                     // Or, pass a TimeStepper object if relevant. Let's assume dt_step is available for unsteady.
    // This requires modifying the base class signature too, or having derived methods.
    // Sticking to current signature and assuming 'time' and maybe an implicit dt are used.
    // A common pattern for unsteady is (I/dt + A) U_new = U_old/dt + F_at_new_time
    // OR (I + dt*A) U_new = U_old + dt*F_at_new_time (Forward Euler)
    // Your Fortran unsteady code uses (Id+dt*A, U, Uk+dt*F) which is U_new = (Id+dt*A)^-1 * (Uk+dt*F).
    // Here U is the current guess, Uk is U_old. So it solves (I+dt*A) U_new = Uk + dt*F_current.
    // The RHS is Uk + dt*F_current, where F_current includes f and BCs at *current* time step t_{k+1}.
    // So, when U_old is provided (Uk), F = Uk + dt*F_from_remplissage.

    F.resize(n);
    F.setZero();

    // Calculate the base RHS vector (f + BCs) as done in Fortran's remplissage_F
    Eigen::VectorXd F_base(n);
    F_base.setZero();

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Add source term f
    for (int k = 0; k < n; ++k) {
        int i, j; // 1-based
        m_grid.indexToCoords(k, i, j);
        double x = j * dx;
        double y = i * dy;
        // Use the appropriate source function based on whether it's a steady or unsteady context
        // If EquationFunctions holds different f_source based on context, use m_eq_funcs.getSource(x, y, time)
        // Otherwise, you might need to pass a flag (is_unsteady) or provide different EquationFunctions objects.
        // Let's assume m_eq_funcs holds the correct f_source for the context.
        F_base(k) = m_eq_funcs.getSource(x, y, time);
    }

    // Add boundary conditions contributions (matching Fortran logic)
    // Note: Assumes BCs are homogeneous (0) on all boundaries except where g or h are non-zero.
    // If g or h represent Dirichlet values, their contributions are added to the RHS.
    // Based on Fortran adding g/(dy^2) and h/(dx^2), this seems to be the case.

    // Top boundary (y=0): nodes 0 to Nx-1 (k=0 to Nx-1)
    for (int k = 0; k < Nx; ++k) {
        int i, j; m_grid.indexToCoords(k, i, j);
        F_base(k) += m_eq_funcs.getG(j * dx, 0.0) / dy2;
    }

    // Bottom boundary (y=Ly): nodes n-Nx to n-1 (k=n-Nx to n-1)
    for (int k = n - Nx; k < n; ++k) {
        int i, j; m_grid.indexToCoords(k, i, j);
        F_base(k) += m_eq_funcs.getG(j * dx, Ly) / dy2;
    }

    // Left boundary (x=0): nodes 0, Nx, 2*Nx, ... (k=0, Nx, 2*Nx, ...)
    for (int k = 0; k < n; k += Nx) {
        int i, j; m_grid.indexToCoords(k, i, j);
        F_base(k) += m_eq_funcs.getH(0.0, i * dy) / dx2;
    }

    // Right boundary (x=Lx): nodes Nx-1, 2*Nx-1, ... (k=Nx-1, 2*Nx-1, ...)
    for (int k = Nx - 1; k < n; k += Nx) {
        int i, j; m_grid.indexToCoords(k, i, j);
        F_base(k) += m_eq_funcs.getH(Lx, i * dy) / dx2;
    }

    // Now, construct the final RHS vector F based on steady or unsteady case
    if (U_old == nullptr) {
        // Steady case: F = F_base
        F = F_base;
        std::cout << "RHS vector F assembled (Steady)" << std::endl;
    } else {
        // Unsteady case (Forward Euler implicit or similar): F = U_old + dt * F_base
        // The dt needs to come from somewhere (e.g., passed as argument)
        // Let's assume dt is available, e.g., passed alongside time
        // Adding dt_step to the signature: assembleRHS(F, time, dt_step, U_old)
        // This requires fixing the base class signature or having different methods.
        // For now, assuming dt_step is available *if* U_old is not null.
        // A better way: the HeatSolver class knows dt and passes it.
        // Let's add dt to the signature and base class for clarity.
        // virtual void assembleRHS(Eigen::VectorXd& F_or_F_local, double time, double dt_step, const Eigen::VectorXd* U_old = nullptr) const = 0;
        // Re-implementing with dt_step in mind:
        // F = *U_old + dt_step * F_base; // Note: Need dt_step

        // FIXING SIGNATURE FOR assembleRHS -> REQUIRES BASE CLASS CHANGE
        // Assuming the signature is: assembleRHS(Eigen::VectorXd& F, double time, double dt_step, const Eigen::VectorXd* U_old = nullptr)
        // This is a significant change to the base class and derived classes.
        // Alternative: Have two separate methods: assembleRHS_steady and assembleRHS_unsteady in derived classes.
        // Let's revert assembleRHS to its original signature and add specific steady/unsteady methods.

        // Reverting assembleRHS signature to: assembleRHS(Eigen::VectorXd& F, double time)
        // And adding specific methods: assembleRHS_steady, assembleRHS_unsteady
        // This is better OOP design anyway.

        // This means the SystemBase/SequentialSystem needs rethinking.
        // Let's define separate methods for steady/unsteady RHS.

        // *** REVISED PLAN ***
        // SystemAssemblerBase:
        // assembleMatrix (pure virtual)
        // assembleRHS_steady (pure virtual)
        // assembleRHS_unsteady (pure virtual)

        // SequentialSystemAssembler:
        // assembleMatrix (override)
        // assembleRHS_steady (override)
        // assembleRHS_unsteady (override)

        // ParallelSystemAssembler:
        // assembleMatrix (override - local)
        // assembleLocalRHS_steady (override)
        // assembleLocalRHS_unsteady (override)
        // parallelMatrixVectorProduct (specific method, not from base)

        // This makes more sense. Okay, let's update the headers first conceptually,
        // but proceed with the *current* assembleRHS implementation assuming it's
        // used *only* for steady (U_old is null). The unsteady logic should go
        // into a separate method or class if the time stepping is complex.

        // --- Assuming this assembleRHS is *only* for the STEADY case ---
         F = F_base; // Final RHS for AU=F steady case
         std::cout << "RHS vector F assembled (Steady)" << std::endl;

        // --- For unsteady, you'd need a different function ---
        // Example for unsteady RHS calculation:
        // void assembleRHS_unsteady(Eigen::VectorXd& RHS, double time, double dt_step, const Eigen::VectorXd& U_old) const {
        //     int n = m_grid.getNumNodes();
        //     Eigen::VectorXd F_base(n); // Calculate f + BC contributions at 'time'
        //     F_base.setZero();
        //     // ... (Fill F_base using m_eq_funcs.getSource(x,y,time), getG/getH) ...
        //     RHS = U_old + dt_step * F_base; // RHS for (I+dt*A)U_new = RHS
        // }
    }
}
```

---

**File: `include/LinearOperatorBase.hpp`**

```cpp
#ifndef LINEAR_OPERATOR_BASE_HPP
#define LINEAR_OPERATOR_BASE_HPP

#include <Eigen/Dense> // Eigen vectors used in multiply method

// Base class interface for a linear operator (matrix-vector multiplication)
// This is what iterative solvers like CG operate on.
class LinearOperatorBase {
public:
    // Virtual destructor
    virtual ~LinearOperatorBase() = default;

    // Pure virtual method for y = Operator * x
    virtual void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const = 0;

    // Optional: Method to get the size of the operator (n x n)
    virtual int size() const = 0;
};

#endif // LINEAR_OPERATOR_BASE_HPP
```

---

**File: `include/MatrixOperator.hpp`**

```cpp
#ifndef MATRIX_OPERATOR_HPP
#define MATRIX_OPERATOR_HPP

#include "LinearOperatorBase.hpp"
#include <Eigen/Sparse> // Typically operates on a sparse matrix

// Class representing a standard matrix as a linear operator
class MatrixOperator : public LinearOperatorBase {
private:
    const Eigen::SparseMatrix<double>& m_A; // Reference to the sparse matrix

public:
    // Constructor takes a const reference to the matrix A
    MatrixOperator(const Eigen::SparseMatrix<double>& A) : m_A(A) {}

    // Implement the multiply method: y = A * x
    void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // Return the size of the operator
    int size() const override { return m_A.rows(); }
};

#endif // MATRIX_OPERATOR_HPP
```

---

**File: `src/MatrixOperator.cpp`**

```cpp
#include "MatrixOperator.hpp"

// Implement the multiply method: y = A * x
void MatrixOperator::multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
    // TODO: Implement matrix-vector product using Eigen's sparse matrix multiplication
    // y = m_A * x;
    y = m_A * x; // Eigen handles this directly for SparseMatrix
}
```

---

**File: `include/UnsteadyOperator.hpp`**

```cpp
#ifndef UNSTEADY_OPERATOR_HPP
#define UNSTEADY_OPERATOR_HPP

#include "LinearOperatorBase.hpp"
#include <Eigen/Sparse> // Typically operates on a sparse matrix

// Class representing the operator (I + dt * A) for the unsteady problem
class UnsteadyOperator : public LinearOperatorBase {
private:
    const Eigen::SparseMatrix<double>& m_A; // Reference to the sparse matrix A
    double m_dt; // Time step value

public:
    // Constructor takes a const reference to matrix A and the time step dt
    UnsteadyOperator(const Eigen::SparseMatrix<double>& A, double dt) : m_A(A), m_dt(dt) {}

    // Implement the multiply method: y = (I + dt * A) * x = x + dt * (A * x)
    void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // Return the size of the operator (same as A's size)
    int size() const override { return m_A.rows(); }
};

#endif // UNSTEADY_OPERATOR_HPP
```

---

**File: `src/UnsteadyOperator.cpp`**

```cpp
#include "UnsteadyOperator.hpp"

// Implement the multiply method: y = (I + dt * A) * x = x + dt * (A * x)
void UnsteadyOperator::multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
    // TODO: Implement (I + dt * A) * x
    // y = x + m_dt * (m_A * x);
    y = x + m_dt * (m_A * x); // Eigen handles sparse multiplication and vector operations
}
```

---

**File: `include/CGSolverBase.hpp`**

```cpp
#ifndef CG_SOLVER_BASE_HPP
#define CG_SOLVER_BASE_HPP

#include <Eigen/Dense> // For vectors
#include <string>      // For output/logging

// Forward declaration
class LinearOperatorBase; // For sequential solver
class ParallelLinearOperatorBase; // For parallel solver

// Base class interface for Conjugate Gradient solvers
// Note: CG algorithms differ significantly between sequential and parallel
// (especially dot products and norms need MPI collectives in parallel).
// A single base class might be tricky unless it's very abstract.
// Let's keep them separate initially and see if a common base makes sense later.

// --- Redefining CG Skeletons without a common base for simplicity ---
// See SequentialCG.hpp and ParallelCG.hpp

#endif // CG_SOLVER_BASE_HPP
```
*(Removing CGSolverBase.hpp from the plan)*

---

**File: `include/SequentialCG.hpp`**

```cpp
#ifndef SEQUENTIAL_CG_HPP
#define SEQUENTIAL_CG_HPP

#include "LinearOperatorBase.hpp" // CG operates on a LinearOperator
#include <Eigen/Dense> // For vectors
#include <iostream>    // For output/logging

// Class implementing the Sequential Conjugate Gradient method
// Solves Op * U = F, where Op is a LinearOperator
class SequentialConjugateGradientSolver {
public:
    // Constructor (no specific state needed yet, could add tolerance/max_iters later)
    SequentialConjugateGradientSolver() = default;

    // Method to solve the system Op * U = F
    // U is the initial guess (input) and solution (output)
    // F is the right-hand side vector (input)
    // op is the linear operator (matrix A or I+dt*A etc.)
    // tolerance is the convergence criterion (e.g., ||residual|| / ||F|| < tolerance)
    // max_iterations is a safety limit
    // Matches the logic of Fortran's Grad_conjuge subroutine.
    void solve(Eigen::VectorXd& U, const Eigen::VectorXd& F, const LinearOperatorBase& op,
               double tolerance = 1e-9, int max_iterations = 10000) const;
};

#endif // SEQUENTIAL_CG_HPP
```

---

**File: `src/SequentialCG.cpp`**

```cpp
#include "SequentialCG.hpp"
#include <cmath>    // For sqrt
#include <iomanip>  // For formatting output

// Method to solve the system Op * U = F using Sequential Conjugate Gradient
void SequentialConjugateGradientSolver::solve(
    Eigen::VectorXd& U, const Eigen::VectorXd& F, const LinearOperatorBase& op,
    double tolerance, int max_iterations) const
{
    // TODO: Implement Sequential Conjugate Gradient algorithm.
    // Use Eigen's vector operations and the 'op.multiply()' method for matrix-vector products.
    // Follow the steps in Fortran's Grad_conjuge subroutine.

    int n = op.size(); // Size of the system

    // Initialize U (input guess, often 0). Ensure it's the correct size.
    if (U.size() != n) U.resize(n);
    // Fortran sequential sets Uk=1 initially for unsteady?? No, Uk is U_old.
    // Parallel stationary sets U=0 initially. Let's assume U is a valid input guess.

    Eigen::VectorXd Gr(n), DIR(n), V(n), Grk(n);
    double alpha, p;

    // Initial Residual: Gr = Op * U - F
    op.multiply(U, Gr); // Gr = Op * U
    Gr = Gr - F;       // Gr = Op * U - F

    // Initial Direction: DIR = -Gr
    DIR = -Gr;

    // Initial norms for convergence check
    double initial_F_norm = F.norm();
    if (initial_F_norm < 1e-12) {
        std::cout << "Sequential CG: RHS vector F is zero, solution is U=0." << std::endl;
        U.setZero();
        return;
    }
    double current_res_norm_sq = Gr.squaredNorm();
    double current_norm_ratio = std::sqrt(current_res_norm_sq) / initial_F_norm;

    std::cout << "Starting Sequential Conjugate Gradient solver..." << std::endl;
    std::cout << std::fixed << std::setprecision(6); // Format output

    int compt = 0; // Iteration counter
    while (current_norm_ratio > tolerance && compt < max_iterations) {

        // V = Op * DIR
        op.multiply(DIR, V);

        // Calculate alpha
        double DIR_dot_V = DIR.dot(V);
        if (DIR_dot_V < 1e-20) { // Check for division by zero or near zero
             std::cerr << "Warning: Sequential CG denominator DIR.dot(V) (" << DIR_dot_V << ") near zero at iteration " << compt << std::endl;
             break; // Stop CG
        }
        alpha = current_res_norm_sq / DIR_dot_V; // alpha = (Gr^T * Gr) / (DIR^T * Op * DIR)

        // Update U: U = U + alpha * DIR
        U = U + alpha * DIR;

        // Update Grk (previous gradient) and Gr (current gradient)
        // Grk = Gr; // Store previous gradient (no explicit copy needed if using Grk=Gr pattern)
        // The Fortran update was Gr = Gr + alpha * V. Let's follow that.
        Grk = Gr; // Need Grk = Gr *before* updating Gr
        Gr = Gr + alpha * V;

        // Calculate beta (using Fletcher-Reeves formula, matching Fortran)
        // p = (Gr^T * Gr) / (Grk^T * Grk)
        double Grk_squared_norm = Grk.squaredNorm();
         if (Grk_squared_norm < 1e-20) { // Check for division by zero or near zero
             // This shouldn't happen unless Grk was already zero, in which case convergence should have occurred earlier.
             std::cerr << "Warning: Sequential CG denominator Grk.squaredNorm() (" << Grk_squared_norm << ") near zero at iteration " << compt << std::endl;
             break; // Stop CG
        }
        p = Gr.squaredNorm() / Grk_squared_norm;

        // Update direction DIR
        DIR = -Gr + p * DIR;

        // Update residual norm squared for next iteration's alpha numerator
        current_res_norm_sq = Gr.squaredNorm();

        // Calculate current norm ratio for convergence check
        current_norm_ratio = std::sqrt(current_res_norm_sq) / initial_F_norm;

        // Output progress (optional)
        if ((compt % 100 == 0 && compt > 0) || current_norm_ratio <= tolerance) {
             std::cout << "CG Iteration " << compt + 1 << ": Residual norm ratio = " << current_norm_ratio << std::endl;
        }

        compt++;
    }

    std::cout << "Sequential Conjugate Gradient solver finished after " << compt << " iterations." << std::endl;
    if (current_norm_ratio <= tolerance) {
        std::cout << "Convergence achieved. Final residual norm ratio: " << current_norm_ratio << std::endl;
    } else {
        std::cout << "Warning: Maximum iterations reached or denominator near zero. Convergence criterion not met. Final residual norm ratio: " << current_norm_ratio << std::endl;
    }
}
```

---

**File: `include/ParallelLinearOperatorBase.hpp`**

```cpp
#ifndef PARALLEL_LINEAR_OPERATOR_BASE_HPP
#define PARALLEL_LINEAR_OPERATOR_BASE_HPP

#include <Eigen/Dense> // For local vectors
#include <mpi.h>       // For MPI_Comm

// Forward declaration
class ParallelSystemAssembler; // Parallel operator likely needs access to assembler's mat-vec product logic

// Base class interface for a Parallel linear operator (distributed matrix-vector multiplication)
class ParallelLinearOperatorBase {
protected:
    // Reference to the parallel system assembler or necessary info for product
    // const ParallelSystemAssembler& m_parallel_assembler; // Or similar mechanism

public:
    // Constructor (needs MPI info, potentially assembler info)
    // ParallelLinearOperatorBase(const ParallelSystemAssembler& assembler) : m_parallel_assembler(assembler) {}

    // Virtual destructor
    virtual ~ParallelLinearOperatorBase() = default;

    // Pure virtual method for y_local = Operator * x_global (or Op * x_local + ghost_exchange)
    // The 'multiply' operation must handle communication for the parallel matrix-vector product.
    // It takes the *local* part of the input vector 'x_local' and the *global* input vector 'x_global' (or needs access to it).
    // It computes and returns the *local* part of the result vector 'y_local'.
    // This requires careful handling of ghost cells or communication within the method.

    // Option 1: Input x_local, output y_local. Communication for global x is implicit.
    // virtual void multiply(const Eigen::VectorXd& x_local, Eigen::VectorXd& y_local) const = 0;

    // Option 2: Input x_global, output y_local. Requires access to global x data structure.
    // This depends heavily on how the parallel vectors are managed.
    // Let's stick closer to the Fortran model where DIR was global but prodMV_para took U and i1:ie.
    // The ParallelSystemAssembler should provide the parallel mat-vec product, and the operator calls it.

    // *** REVISED PLAN ***
    // The ParallelSystemAssembler class will *hold* the local matrix and provide the
    // `parallelMatrixVectorProduct` method. The Parallel CG solver will call *that* method.
    // A separate `ParallelLinearOperator` might be overly complex if it just wraps the assembler's method.
    // Let's pass the ParallelSystemAssembler (or relevant parts) directly to the ParallelCG solver.

    // If we *do* keep a ParallelLinearOperator, it would likely store the local matrix
    // and handle the communication itself, potentially needing MPI info.
    // Example with local matrix and MPI comm:
    // virtual void multiply(const Eigen::VectorXd& x_local, Eigen::VectorXd& y_local, MPI_Comm comm, const GridGeometry& grid, int i1, int ie) const = 0;
    // This gets complicated fast.

    // Let's simplify: ParallelCG knows about MPI and the ParallelSystemAssembler.
    // Remove ParallelLinearOperatorBase from the plan for now.
};

#endif // PARALLEL_LINEAR_OPERATOR_BASE_HPP
```
*(Removing ParallelLinearOperatorBase.hpp, ParallelMatrixOperator.hpp, ParallelUnsteadyOperator.hpp from the plan)*

---

**File: `include/ParallelSystem.hpp`**

```cpp
#ifndef PARALLEL_SYSTEM_HPP
#define PARALLEL_SYSTEM_HPP

// Note: This class might not inherit from SystemBase anymore if assembleMatrix signature changes.
// Let's redefine it to hold local data and provide parallel operations.

#include "GridGeometry.hpp"
#include "PhysicsParameters.hpp"
#include "EquationFunctions.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense> // For local vectors
#include <mpi.h>       // For MPI_Comm

// Class to manage the parallel system (local matrix, local RHS, parallel mat-vec product)
class ParallelSystem {
private:
    MPI_Comm m_comm;         // MPI Communicator
    int m_rank;              // MPI Rank (Me)
    int m_size;              // MPI Size (Np)

    const GridGeometry& m_grid;       // Reference to grid info
    const PhysicsParameters& m_physics; // Reference to physics params
    const EquationFunctions& m_eq_funcs; // Reference to equation functions

    int m_i1; // Start index (0-based) for this process's portion
    int m_ie; // End index (0-based) for this process's portion
    int m_local_n; // Number of local nodes (ie - i1 + 1)

    Eigen::SparseMatrix<double> m_A_local; // Local portion of the matrix A

public:
    // Constructor
    ParallelSystem(const GridGeometry& grid, const PhysicsParameters& physics,
                   const EquationFunctions& eq_funcs, MPI_Comm comm);

    // Accessor methods for local info
    int getLocalStartIndex() const { return m_i1; }
    int getLocalEndIndex() const { return m_ie; }
    int getLocalNumNodes() const { return m_local_n; }
    int getGlobalNumNodes() const { return m_grid.getNumNodes(); }
    int getRank() const { return m_rank; }
    int getSize() const { return m_size; }
    MPI_Comm getComm() const { return m_comm; }
    const Eigen::SparseMatrix<double>& getLocalMatrix() const { return m_A_local; }


    // Method to determine the local workload range (i1, ie)
    // Matches Fortran's 'charge' subroutine logic.
    void determineLocalRange();

    // Method to assemble the local portion of the matrix A
    // Implement logic from Fortran's parallel matrix filling
    void assembleLocalMatrix();

    // Method to assemble the local portion of the RHS vector F
    // Implement logic from Fortran's mod_remplissage or mod_remplissage_instationnaire
    // time: current time for unsteady source/BCs
    // dt_step: time step size (needed for unsteady RHS = U_old + dt*F)
    // U_old_local: local part of U_old vector (needed for unsteady RHS)
    void assembleLocalRHS_steady(Eigen::VectorXd& F_local, double time) const;
    void assembleLocalRHS_unsteady(Eigen::VectorXd& F_local, double time, double dt_step, const Eigen::VectorXd& U_old_global) const; // Needs U_old_global for access to ghost values

    // Method for the parallel matrix-vector product y_local = A_local * x_global
    // This method handles the communication for ghost cells (neighboring values of x needed).
    // Implement logic from Fortran's prodMV_para, including MPI_Sendrecv for ghost cells.
    void parallelMatrixVectorProduct(const Eigen::VectorXd& x_global, Eigen::VectorXd& y_local) const;

    // For parallel CG, need parallel dot product and norm (MPI_Allreduce).
    // These can be standalone functions or methods of ParallelCG/ParallelSystem.
};

#endif // PARALLEL_SYSTEM_HPP
```

---

**File: `src/ParallelSystem.cpp`**

```cpp
#include "ParallelSystem.hpp"
#include "GridGeometry.hpp"
#include "PhysicsParameters.hpp"
#include "EquationFunctions.hpp"
#include <cmath>
#include <vector> // For MPI send/recv buffers
#include <iostream>

// Constructor
ParallelSystem::ParallelSystem(const GridGeometry& grid, const PhysicsParameters& physics,
                               const EquationFunctions& eq_funcs, MPI_Comm comm)
    : m_comm(comm), m_grid(grid), m_physics(physics), m_eq_funcs(eq_funcs)
{
    // Get rank and size immediately
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);

    // Determine local range
    determineLocalRange();
    m_local_n = m_ie - m_i1 + 1;

    if (m_rank == 0) {
         std::cout << "Parallel System Initialized with " << m_size << " processes." << std::endl;
         std::cout << "Each process handles " << m_local_n << " nodes (approx)." << std::endl;
    }
    // Print range for each process
    // std::cout << "Rank " << m_rank << " handles global indices [" << m_i1 << ", " << m_ie << "]" << std::endl;

    // Allocate the local matrix
    // m_A_local.resize(m_local_n, m_grid.getNumNodes()); // Local rows, global columns (for multiply)
    // Or more commonly, store only local-to-local and local-to-ghost connections.
    // Eigen::SparseMatrix can handle this, but efficient assembly requires careful indexing.
    // Let's resize A_local for local_n rows and local_n columns plus ghost connections.
    // A simpler approach is to resize for global columns but only fill local rows.
    m_A_local.resize(m_local_n, m_grid.getNumNodes());
     // Estimate non-zeros per row (5-point stencil)
    m_A_local.reserve(Eigen::VectorXi::Constant(m_local_n, 5));
}

// Method to determine the local workload range (i1, ie)
// Matches Fortran's 'charge' subroutine logic (0-based output).
void ParallelSystem::determineLocalRange() {
    // TODO: Implement logic from Fortran's 'charge'
    // Given total nodes 'n' and total processes 'Np'.
    // Calculate base size per process and remainder.
    // Assign ranges [i1, ie] for each process Me.
    int n = m_grid.getNumNodes();
    int Np = m_size;
    int Me = m_rank;

    int base_size = n / Np;
    int remainder = n % Np;

    if (Me < remainder) {
        m_i1 = Me * (base_size + 1);
        m_ie = m_i1 + base_size;
    } else {
        m_i1 = Me * base_size + remainder;
        m_ie = m_i1 + base_size - 1;
    }
    // m_i1 and m_ie are 0-based global indices.
}

// Method to assemble the local portion of the matrix A
// Implement logic from Fortran's parallel matrix filling (A(1:5, i1:ie))
void ParallelSystem::assembleLocalMatrix() {
    // TODO: Implement local matrix assembly.
    // Iterate through global indices k from m_i1 to m_ie.
    // For each k, calculate matrix entries for A(k, neighbor_k).
    // Only insert entries where the row index is local (k), but column index (neighbor_k) can be global.
    // The Fortran A(1:5, i1:ie) structure suggests storing only the 5 bands relative to the local index,
    // but the prodMV_para function uses Xp(i-Nx:i+Nx) which is global.
    // Eigen::SparseMatrix allows inserting global columns.

    m_A_local.setZero(); // Ensure it's clear before assembly

    double dx = m_grid.getDx();
    double dy = m_grid.getDy();
    double D = m_physics.D;
    int Nx = m_grid.getNx();
    int Ny = m_grid.getNy(); // Needed for boundary checks
    int n = m_grid.getNumNodes(); // Needed for neighbor index bounds

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Loop through local rows (corresponding to global indices k from m_i1 to m_ie)
    for (int local_k = 0; local_k < m_local_n; ++local_k) {
        int k = m_i1 + local_k; // Global index

        int i, j; // 1-based grid coordinates
        m_grid.indexToCoords(k, i, j); // Get 1-based i, j

        // Diagonal entry (global column k)
        m_A_local.insert(local_k, k) = 2.0 * D * (1.0/dx2 + 1.0/dy2);

        // Off-diagonal entries for neighbors (global column neighbor_k)

        // Neighbor to the left (k-1): exists if j > 1
        if (j > 1) {
            m_A_local.insert(local_k, k - 1) = -D / dx2;
        }

        // Neighbor to the right (k+1): exists if j < Nx
        if (j < Nx) {
             m_A_local.insert(local_k, k + 1) = -D / dx2;
        }

        // Neighbor above (k-Nx): exists if i > 1
        if (i > 1) {
            m_A_local.insert(local_k, k - Nx) = -D / dy2;
        }

        // Neighbor below (k+Nx): exists if i < Ny
        if (i < Ny) {
            m_A_local.insert(local_k, k + Nx) = -D / dy2;
        }
    }

    m_A_local.makeCompressed(); // Optimize storage

    // std::cout << "Rank " << m_rank << ": Local matrix assembled (" << m_local_n << "x" << m_grid.getNumNodes() << ")" << std::endl;
}


// Method to assemble the local portion of the RHS vector F (Steady)
// Implement logic from Fortran's mod_remplissage, iterating only through local indices.
// time: current time for source/BCs (though steady F doesn't depend on time or U_old)
void ParallelSystem::assembleLocalRHS_steady(Eigen::VectorXd& F_local, double time) const {
    // TODO: Implement local RHS assembly for steady case.
    // Calculate F_local for global indices k from m_i1 to m_ie.
    // F_local has size m_local_n. F_local(local_k) corresponds to global index k = m_i1 + local_k.

    F_local.resize(m_local_n);
    F_local.setZero();

    double dx = m_grid.getDx();
    double dy = m_grid.getDy();
    double Lx = m_grid.getLx(); // Needed for getH
    double Ly = m_grid.getLy(); // Needed for getG
    int Nx = m_grid.getNx();
    int Ny = m_grid.getNy();

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Loop through local nodes (0-based local index 'local_k')
    for (int local_k = 0; local_k < m_local_n; ++local_k) {
        int k = m_i1 + local_k; // Corresponding global index (0-based)

        int i, j; // 1-based grid coordinates
        m_grid.indexToCoords(k, i, j); // Get 1-based i, j

        double x = j * dx; // X-coordinate
        double y = i * dy; // Y-coordinate

        // Add source term contribution f
        // Use the stationary source function here (or the general getSource with appropriate time)
        F_local(local_k) = m_eq_funcs.getSource(x, y, time); // Note: time might be irrelevant for steady f1

        // Add boundary conditions contributions *if* this local node is adjacent to a boundary
        // Check global index k or grid coordinates i, j

        // Top boundary (y=0): Node (j,i=1) is adjacent to y=0. Global indices k=0..Nx-1.
        if (i == 1) { // If this row is the first row
             F_local(local_k) += m_eq_funcs.getG(x, 0.0) / dy2;
        }

        // Bottom boundary (y=Ly): Node (j,i=Ny) is adjacent to y=Ly. Global indices k=n-Nx..n-1.
        if (i == Ny) { // If this row is the last row
             F_local(local_k) += m_eq_funcs.getG(x, Ly) / dy2;
        }

        // Left boundary (x=0): Node (j=1,i) is adjacent to x=0. Global indices k=0, Nx, 2Nx, ...
        if (j == 1) { // If this column is the first column
            F_local(local_k) += m_eq_funcs.getH(0.0, y) / dx2;
        }

        // Right boundary (x=Lx): Node (j=Nx,i) is adjacent to x=Lx. Global indices k=Nx-1, 2Nx-1, ...
        if (j == Nx) { // If this column is the last column
            F_local(local_k) += m_eq_funcs.getH(Lx, y) / dx2;
        }
    }
    // std::cout << "Rank " << m_rank << ": Local RHS (Steady) assembled." << std::endl;
}

// Method to assemble the local portion of the RHS vector F (Unsteady)
// This assumes Forward Euler: (I + dt*A)U_new = U_old + dt*F_base(t_new)
// RHS = U_old + dt_step * F_base(t_new)
// U_old_global: The full U_old vector (needed to access values outside local range).
// time: The *new* time t_{k+1} for evaluating f_base.
// dt_step: The time step size dt.
void ParallelSystem::assembleLocalRHS_unsteady(Eigen::VectorXd& F_local, double time, double dt_step, const Eigen::VectorXd& U_old_global) const {
     // TODO: Implement local RHS assembly for unsteady case.
     // Calculate F_local for global indices k from m_i1 to m_ie.
     // F_local = U_old_local + dt_step * F_base_local(time).
     // U_old_local is the part of U_old_global from m_i1 to m_ie.

    F_local.resize(m_local_n);
    F_local.setZero();

    double dx = m_grid.getDx();
    double dy = m_grid.getDy();
    double Lx = m_grid.getLx();
    double Ly = m_grid.getLy();
    int Nx = m_grid.getNx();
    int Ny = m_grid.getNy();

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    // Calculate the local base RHS vector (f + BCs) at the new time 'time'
    Eigen::VectorXd F_base_local(m_local_n);
    F_base_local.setZero();

     // Loop through local nodes
    for (int local_k = 0; local_k < m_local_n; ++local_k) {
        int k = m_i1 + local_k; // Global index (0-based)
        int i, j; // 1-based
        m_grid.indexToCoords(k, i, j);

        double x = j * dx;
        double y = i * dy;

        // Add source term contribution f at the new time
        F_base_local(local_k) = m_eq_funcs.getSource(x, y, time); // Use the unsteady source function

        // Add boundary conditions contributions *if* this local node is adjacent to a boundary
        if (i == 1) { F_base_local(local_k) += m_eq_funcs.getG(x, 0.0) / dy2; }
        if (i == Ny) { F_base_local(local_k) += m_eq_funcs.getG(x, Ly) / dy2; }
        if (j == 1) { F_base_local(local_k) += m_eq_funcs.getH(0.0, y) / dx2; }
        if (j == Nx) { F_base_local(local_k) += m_eq_funcs.getH(Lx, y) / dx2; }
    }

    // Construct the final local RHS vector: U_old_local + dt_step * F_base_local
    // Need to access U_old_global from indices m_i1 to m_ie. Eigen slicing helps here.
    // F_local = U_old_global.segment(m_i1, m_local_n) + dt_step * F_base_local;
    F_local = U_old_global.segment(m_i1, m_local_n) + dt_step * F_base_local;


    // std::cout << "Rank " << m_rank << ": Local RHS (Unsteady) assembled." << std::endl;
}


// Method for the parallel matrix-vector product y_local = A_local * x_global
// This method handles the communication for ghost cells (neighboring values of x needed).
// Implement logic from Fortran's prodMV_para, including MPI_Sendrecv for ghost cells.
// x_global: The full global vector x (needed for accessing ghost values).
// y_local: The computed local part of y = A * x (output).
void ParallelSystem::parallelMatrixVectorProduct(const Eigen::VectorXd& x_global, Eigen::VectorXd& y_local) const {
    // TODO: Implement parallel matrix-vector product.
    // y_local.resize(m_local_n); // Ensure y_local is correct size
    // y_local.setZero();

    // In prodMV_para, the calculation for y(i) depends on x(i-Nx), x(i-1), x(i), x(i+1), x(i+Nx).
    // For a local row 'i' (global index), x(i) is local, but neighbors like x(i-1) or x(i+Nx) might be on other processes.
    // The Fortran prodMV_para received the full X vector but only computed y for i1:ie.
    // Eigen::SparseMatrix::operator*() can work if A_local stores columns corresponding to global indices.
    // y_local = m_A_local * x_global;

    // However, if x_global is very large, passing it everywhere might be inefficient.
    // A more standard parallel sparse mat-vec:
    // 1. y_local = (local diagonal/block of A) * x_local
    // 2. Identify off-diagonal connections in A_local that point to *remote* x values (ghost cells).
    // 3. Communicate ghost cell values from neighbors into a local buffer.
    // 4. y_local += (local off-diagonal blocks of A) * x_ghost_buffer

    // Given that your Fortran prodMV_para receives the *entire* X vector (though allocated locally as Xp(1-Nx:n+Nx) and filled globally),
    // the easiest translation is to require the caller to provide the full x_global vector.
    // If x_global is properly communicated or gathered *before* this function call, then the Eigen multiplication works directly.
    // If x_global isn't available globally on each process, this function must handle the communication.
    // The Fortran `MPI_SENDRECV` in `Grad_conjuge_para` suggests ghost cell exchange *before* `prodMV_para`.
    // This means the `DIR` vector in Fortran's `Grad_conjuge_para` is *not* fully global but has neighboring ghost values added before `prodMV_para` is called.

    // Let's assume the caller (ParallelCG) handles ghost cell communication for the input vector 'x'
    // *before* calling this method, placing ghost values into `x_global`. This is inefficient
    // if x_global is truly global, but matches the Fortran call pattern.

    // Efficient parallel mat-vec needs ghost exchange *within* the method or managed by the caller.
    // Re-reading Fortran: DIR is size n. MPI_SENDRECV updates DIR(i1-Nx:i1-1) and DIR(ie+1:ie+Nx).
    // Then V=prodMV_para(A,DIR,n,Nx,i1,ie) uses the *local* part of A (i1:ie) and this *partially updated global* DIR.
    // This means DIR needs to be a global vector that gets portions updated by neighbors.

    // Let's structure `parallelMatrixVectorProduct` to receive the *full* global input vector `x_global`
    // and compute the *local* result `y_local`. Eigen handles the sparsity lookup.

    y_local.resize(m_local_n); // Ensure output vector is correct size
    y_local.setZero();

    // Perform the local part of y = A * x
    // Eigen's sparse matrix multiplication handles this directly if A_local is defined
    // with local rows and global columns, and x_global is the full vector.
    // Iterate through local rows (0-based local index 'local_k')
    for (int local_k = 0; local_k < m_local_n; ++local_k) {
        // global_k = m_i1 + local_k;
        // y_local(local_k) = sum_{j=0}^{n-1} A_local(local_k, j) * x_global(j)
        // Eigen handles this loop and dot product implicitly with operator*.
        // The result of m_A_local * x_global is a vector of size m_local_n.
    }
    y_local = m_A_local * x_global; // Eigen handles the sparse multiplication

    // This assumes x_global is available on each process.
    // If x_global is distributed and only ghost values are exchanged, the multiply method would be more complex.
    // Let's stick to the direct Eigen multiply assuming x_global is accessible (potentially replicated or using advanced Eigen features not shown here).
    // A more MPI-idiomatic approach would be:
    // 1. Exchange necessary ghost cells of x_local.
    // 2. Compute y_local = A_local_local * x_local + A_local_ghost * x_ghost.
    // This requires splitting A_local and managing ghost buffers explicitly.

    // For this skeleton, let's use the simplest interpretation based on Fortran's `prodMV_para(A,X,n,Nx,i1,ie)`
    // where A has local rows (i1:ie), X is the full global vector (conceptually), and it computes local result (i1:ie).
    // `m_A_local` has `m_local_n` rows and `n` columns.
    // `x_global` is size `n`.
    // The result `m_A_local * x_global` is size `m_local_n`. This matches `y_local`.
    // So, `y_local = m_A_local * x_global;` is the correct Eigen implementation *if* x_global is the full vector.

    // std::cout << "Rank " << m_rank << ": Performed parallel matrix-vector product." << std::endl;
}

// Optional: Implement parallel dot product and norm if needed outside ParallelCG
// These would use MPI_Allreduce
/*
double ParallelSystem::parallelDotProduct(const Eigen::VectorXd& v1_local, const Eigen::VectorXd& v2_local) const {
    double local_dot = v1_local.dot(v2_local);
    double global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    return global_dot;
}

double ParallelSystem::parallelNorm(const Eigen::VectorXd& v_local) const {
     double local_sq_norm = v_local.squaredNorm();
     double global_sq_norm;
     MPI_Allreduce(&local_sq_norm, &global_sq_norm, 1, MPI_DOUBLE, MPI_SUM, m_comm);
     return std::sqrt(global_sq_norm);
}
*/
```

---

**File: `include/ParallelCG.hpp`**

```cpp
#ifndef PARALLEL_CG_HPP
#define PARALLEL_CG_HPP

#include "ParallelSystem.hpp" // Parallel CG needs access to parallel mat-vec
#include <Eigen/Dense>       // For local vectors
#include <mpi.h>             // For MPI

// Class implementing the Parallel Conjugate Gradient method
// Solves Op * U = F where Op is a parallel linear operator (provided by ParallelSystem)
class ParallelConjugateGradientSolver {
private:
    MPI_Comm m_comm; // MPI Communicator
    int m_rank;      // MPI Rank
    int m_size;      // MPI Size

    // Reference to the ParallelSystem assembler to get the parallel matrix-vector product
    const ParallelSystem& m_parallel_system;

public:
    // Constructor
    ParallelConjugateGradientSolver(const ParallelSystem& parallel_system, MPI_Comm comm);

    // Method to solve the system Op * U_global = F_global
    // U_global is the initial guess (input) and solution (output) - needs to be managed globally
    // F_global is the right-hand side vector (input) - also managed globally
    // This CG solver will operate primarily on *local* vector parts,
    // performing MPI communication for dot products, norms, and matrix-vector products.
    // It updates the *local* part of U_global.
    // Note: Requires U_global and F_global to be accessible by all processes.
    // A more advanced approach might manage U and F as distributed vectors.
    void solve(Eigen::VectorXd& U_global, const Eigen::VectorXd& F_global,
               double tolerance = 1e-4, int max_iterations = 10000) const;

private:
    // Helper functions for parallel vector operations using MPI_Allreduce
    double parallelDotProduct(const Eigen::VectorXd& v1_local, const Eigen::VectorXd& v2_local) const;
    double parallelNorm(const Eigen::VectorXd& v_local) const;
};

#endif // PARALLEL_CG_HPP
```

---

**File: `src/ParallelCG.cpp`**

```cpp
#include "ParallelCG.hpp"
#include <cmath>    // For sqrt
#include <iomanip>  // For formatting output
#include <iostream> // For output

// Constructor
ParallelConjugateGradientSolver::ParallelConjugateGradientSolver(const ParallelSystem& parallel_system, MPI_Comm comm)
    : m_comm(comm), m_parallel_system(parallel_system)
{
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);
}

// Method to solve the system Op * U_global = F_global using Parallel Conjugate Gradient
void ParallelConjugateGradientSolver::solve(
    Eigen::VectorXd& U_global, const Eigen::VectorXd& F_global,
    double tolerance, int max_iterations) const
{
    // TODO: Implement Parallel Conjugate Gradient algorithm.
    // This is similar to the sequential version, but dot products and norms
    // must use MPI_Allreduce, and matrix-vector products must use the parallel method.
    // Follow the steps in Fortran's Grad_conjuge_para subroutine.

    int n = m_parallel_system.getGlobalNumNodes();
    int local_n = m_parallel_system.getLocalNumNodes();
    int i1 = m_parallel_system.getLocalStartIndex();
    int ie = m_parallel_system.getLocalEndIndex();

    // Ensure U_global is the correct size
     if (U_global.size() != n) U_global.resize(n);
     // Ensure F_global is the correct size
     if (F_global.size() != n) F_global.resize(n);


    // Local vector parts
    Eigen::VectorXd Gr_local(local_n), DIR_local(local_n), V_local(local_n), Grk_local(local_n);

    // Initialize U_global (input guess, often 0). The caller should set this.
    // Get the local part of U
    Eigen::VectorXd U_local = U_global.segment(i1, local_n);
    Eigen::VectorXd F_local = F_global.segment(i1, local_n); // Get the local part of F

    // Initial Residual: Gr = Op * U - F
    // Need to perform Op * U_global first, then subtract F_local.
    // The Op * U_global result is local.
    // This requires a way to get Op * U_global result locally.
    // The ParallelSystem::parallelMatrixVectorProduct(x_global, y_local) does exactly this.
    m_parallel_system.parallelMatrixVectorProduct(U_global, Gr_local); // Gr_local = Op * U_global (local part)
    Gr_local = Gr_local - F_local;                                     // Gr_local = Op * U_global - F_local

    // Initial Direction: DIR_local = -Gr_local
    DIR_local = -Gr_local;
    // The full DIR_global vector is needed for the next matrix-vector product.
    // We need to copy DIR_local back into the global DIR_global vector.
    Eigen::VectorXd DIR_global = Eigen::VectorXd::Zero(n); // Allocate full global DIR
    DIR_global.segment(i1, local_n) = DIR_local; // Copy local part

    // Initial norms for convergence check (using parallel norm)
    double initial_F_norm = parallelNorm(F_local); // Global norm of F
    if (initial_F_norm < 1e-12) {
        if (m_rank == 0) std::cout << "Parallel CG: Global RHS vector F is zero, solution is U=0." << std::endl;
        U_global.setZero(); // Set global U to zero
        return;
    }
    double current_res_norm_sq = parallelDotProduct(Gr_local, Gr_local); // Global squared norm of Gr
    double current_norm_ratio = std::sqrt(current_res_norm_sq) / initial_F_norm;

    if (m_rank == 0) {
        std::cout << "Starting Parallel Conjugate Gradient solver..." << std::endl;
        std::cout << std::fixed << std::setprecision(6); // Format output
    }

    int compt = 0; // Iteration counter
    while (current_norm_ratio > tolerance && compt < max_iterations) {

        // V_local = Op * DIR_global
        // This step requires the full DIR_global vector on each process or ghost exchange.
        // The ParallelSystem::parallelMatrixVectorProduct method handles this using DIR_global.
        m_parallel_system.parallelMatrixVectorProduct(DIR_global, V_local); // V_local = Op * DIR_global (local part)


        // Calculate alpha
        double global_Gr_squared_norm = current_res_norm_sq; // Already computed global norm
        double global_DIR_dot_V = parallelDotProduct(DIR_local, V_local);

        if (global_DIR_dot_V < 1e-20) { // Check for division by zero or near zero
             if (m_rank == 0) std::cerr << "Warning: Parallel CG denominator DIR.dot(V) (" << global_DIR_dot_V << ") near zero at iteration " << compt << std::endl;
             break; // Stop CG
        }
        alpha = global_Gr_squared_norm / global_DIR_dot_V; // alpha = (Gr^T * Gr) / (DIR^T * Op * DIR)

        // Update U_global (only update the local part U_local)
        U_local = U_local + alpha * DIR_local.segment(i1, local_n); // Need local part of DIR_global here
                                                                 // But DIR_local was computed from Gr_local, so it *is* the local part of DIR_global.
        U_local = U_local + alpha * DIR_local;

        // Update Grk_local (previous gradient) and Gr_local (current gradient)
        Grk_local = Gr_local; // Store previous local gradient
        Gr_local = Gr_local + alpha * V_local; // Update local gradient

        // Calculate beta (using Fletcher-Reeves formula, matching Fortran)
        // p = (Gr^T * Gr) / (Grk^T * Grk)
        double global_Grk_squared_norm = parallelDotProduct(Grk_local, Grk_local);
         if (global_Grk_squared_norm < 1e-20) { // Check for division by zero or near zero
             if (m_rank == 0) std::cerr << "Warning: Parallel CG denominator Grk.squaredNorm() (" << global_Grk_squared_norm << ") near zero at iteration " << compt << std::endl;
             break; // Stop CG
        }
        p = parallelDotProduct(Gr_local, Gr_local) / global_Grk_squared_norm;

        // Update direction DIR_local -> Need to update DIR_global based on this
        DIR_local = -Gr_local + p * DIR_local; // Update local part of DIR
        DIR_global.segment(i1, local_n) = DIR_local; // Copy local part back into global DIR for the next matrix-vector product


        // Update residual norm squared for next iteration's alpha numerator
        current_res_norm_sq = parallelDotProduct(Gr_local, Gr_local); // Recalculate global squared norm of Gr

        // Calculate current norm ratio for convergence check
        current_norm_ratio = std::sqrt(current_res_norm_sq) / initial_F_norm;

        // Output progress (optional) - only rank 0 prints
        if (m_rank == 0) {
             if ((compt % 100 == 0 && compt > 0) || current_norm_ratio <= tolerance) {
                  std::cout << "CG Iteration " << compt + 1 << ": Residual norm ratio = " << current_norm_ratio << std::endl;
             }
        }

        compt++;
    }

    // Copy the final local solution U_local back into the global U_global
    U_global.segment(i1, local_n) = U_local;

    if (m_rank == 0) {
        std::cout << "Parallel Conjugate Gradient solver finished after " << compt << " iterations." << std::endl;
        if (current_norm_ratio <= tolerance) {
            std::cout << "Convergence achieved. Final residual norm ratio: " << current_norm_ratio << std::endl;
        } else {
            std::cout << "Warning: Maximum iterations reached or denominator near zero. Convergence criterion not met. Final residual norm ratio: " << current_norm_ratio << std::endl;
        }
    }
}

// Helper functions for parallel vector operations using MPI_Allreduce
double ParallelConjugateGradientSolver::parallelDotProduct(const Eigen::VectorXd& v1_local, const Eigen::VectorXd& v2_local) const {
    // TODO: Implement parallel dot product using MPI_Allreduce
    double local_dot = v1_local.dot(v2_local);
    double global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    return global_dot;
}

double ParallelConjugateGradientSolver::parallelNorm(const Eigen::VectorXd& v_local) const {
    // TODO: Implement parallel norm (global sqrt of global squared norm) using MPI_Allreduce
    double local_sq_norm = v_local.squaredNorm();
    double global_sq_norm;
    MPI_Allreduce(&local_sq_norm, &global_sq_norm, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    return std::sqrt(global_sq_norm);
}
```

---

**File: `include/SequentialHeatSolver.hpp`**

```cpp
#ifndef SEQUENTIAL_HEAT_SOLVER_HPP
#define SEQUENTIAL_HEAT_SOLVER_HPP

#include "GridGeometry.hpp"
#include "PhysicsParameters.hpp"
#include "EquationFunctions.hpp"
#include "SequentialSystem.hpp"
#include "SequentialCG.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <string>
#include <vector>

// Class to orchestrate the sequential solving process (steady or unsteady)
class SequentialHeatSolver {
private:
    // Problem configuration (passed in constructor)
    GridGeometry m_grid;
    PhysicsParameters m_physics;
    EquationFunctions m_eq_funcs;

    // System components (managed by the solver)
    SequentialSystemAssembler m_assembler;
    SequentialConjugateGradientSolver m_cg_solver;

    // Data for the linear system and solution
    Eigen::SparseMatrix<double> m_A; // Matrix A for Au=F or (I+dtA)u_new=...
    Eigen::VectorXd m_U;             // Solution vector
    Eigen::VectorXd m_F;             // RHS vector for the linear system solver

public:
    // Constructor takes problem definition details
    SequentialHeatSolver(int Nx, int Ny, double Lx, double Ly, double D,
                         std::function<double(double x, double y, double t)> f_source_func,
                         std::function<double(double x, double y)> g_func,
                         std::function<double(double x, double y)> h_func);

    // Method to run the steady-state solver
    void runSteady();

    // Method to run the unsteady solver
    // nt: number of time steps
    // dt: time step size
    void runUnsteady(int nt, double dt);

private:
    // Helper method to save the solution vector to files
    void saveResults(const std::string& filename_U, const std::string& filename_Ub) const;

    // Helper method to calculate the operator for CG (A for steady, I+dt*A for unsteady)
    // Returns a unique_ptr to avoid slicing with base class pointers.
    std::unique_ptr<LinearOperatorBase> createLinearOperator(double dt_step_for_unsteady = 0.0) const;
};

#endif // SEQUENTIAL_HEAT_SOLVER_HPP
```

---

**File: `src/SequentialHeatSolver.cpp`**

```cpp
#include "SequentialHeatSolver.hpp"
#include "MatrixOperator.hpp"   // Needed for steady case
#include "UnsteadyOperator.hpp" // Needed for unsteady case
#include <fstream>              // For file output
#include <iomanip>              // For formatting output
#include <iostream>             // For console output
#include <chrono>               // For timing

// Constructor
SequentialHeatSolver::SequentialHeatSolver(
    int Nx, int Ny, double Lx, double Ly, double D,
    std::function<double(double x, double y, double t)> f_source_func,
    std::function<double(double x, double y)> g_func,
    std::function<double(double x, double y)> h_func)
    : m_grid(Nx, Ny, Lx, Ly),       // Initialize GridGeometry
      m_physics(D),                 // Initialize PhysicsParameters
      m_eq_funcs(f_source_func, g_func, h_func), // Initialize EquationFunctions
      m_assembler(m_grid, m_physics, m_eq_funcs) // Initialize Assembler
      // CG solver is default constructed
{
    // Allocate U and F vectors based on grid size
    int n = m_grid.getNumNodes();
    m_U.resize(n);
    m_F.resize(n);

    // Initial guess for U (often zero)
    m_U.setZero();

    // The matrix A is typically constant for both steady and unsteady (explicit time stepping)
    // Assemble the matrix once during setup
    std::cout << "Assembling matrix A (Sequential)..." << std::endl;
    m_assembler.assembleMatrix(m_A);
    std::cout << "Matrix A assembled." << std::endl;
}

// Helper method to calculate the operator for CG (A for steady, I+dt*A for unsteady)
std::unique_ptr<LinearOperatorBase> SequentialHeatSolver::createLinearOperator(double dt_step_for_unsteady) const {
    if (dt_step_for_unsteady == 0.0) {
        // Steady operator is just A
        return std::make_unique<MatrixOperator>(m_A);
    } else {
        // Unsteady operator is (I + dt*A)
        return std::make_unique<UnsteadyOperator>(m_A, dt_step_for_unsteady);
    }
}


// Method to run the steady-state solver
void SequentialHeatSolver::runSteady() {
    std::cout << "Running sequential steady-state solver..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Assemble the RHS vector F for the steady case (time=0.0, U_old=nullptr)
    // Assuming assembleRHS method handles steady logic when U_old is null.
    // Let's pass time = 0.0 for steady RHS calculation.
    // Need to adjust SequentialSystem::assembleRHS or call a dedicated assembleRHS_steady
    // Based on the SystemBase/SequentialSystem redesign, assembleRHS_steady should be used.
    // Let's assume SequentialSystem has assembleRHS_steady(F).
    // This requires changing SystemBase and SequentialSystem.

    // *** REVISED PLAN (AGAIN) ***
    // SystemAssemblerBase: assembleMatrix, assembleRHS_steady, assembleRHS_unsteady
    // SequentialSystemAssembler: assembleMatrix, assembleRHS_steady, assembleRHS_unsteady (overrides)
    // ParallelSystemAssembler: assembleLocalMatrix, assembleLocalRHS_steady, assembleLocalRHS_unsteady (overrides)
    // Use these specific methods in the solver classes.

    // Re-implementing runSteady assuming assembleRHS_steady exists:
    std::cout << "Assembling RHS vector F (Steady)..." << std::endl;
    m_assembler.assembleRHS_steady(m_F, 0.0); // Use assembleRHS_steady, time might be 0 or arbitrary for steady

    // 2. Create the linear operator (Matrix A)
    // Need to create the operator specific to the system being solved.
    // For steady: AU=F, operator is A.
    // For unsteady: (I+dtA)U_new = U_old + dtF, operator is (I+dtA).
    // Create operator here, pass to CG.

    // For steady AU=F, the operator is simply MatrixOperator(m_A).
    MatrixOperator linear_operator(m_A);

    // 3. Solve the linear system AU = F using CG
    // Pass m_U as initial guess (already zeroed), m_F as RHS, and the operator.
    std::cout << "Solving AU=F using CG..." << std::endl;
    m_cg_solver.solve(m_U, m_F, linear_operator, 1e-9, 10000); // Use appropriate tolerance for steady

    // 4. Save the final solution
    saveResults("VecteurU_steady", "VecteurU_b_steady"); // Use specific filenames for steady

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Sequential steady-state solver finished in " << elapsed.count() << " seconds." << std::endl;
}


// Method to run the unsteady solver (explicit time stepping)
// Solves (I + dt*A) U_new = U_old + dt*F(t_new) at each time step.
// nt: number of time steps
// dt: time step size
void SequentialHeatSolver::runUnsteady(int nt, double dt) {
    std::cout << "Running sequential unsteady solver for " << nt << " steps with dt=" << dt << "..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize U_old (initial condition - often 0)
    Eigen::VectorXd U_old = m_U; // m_U is currently the initial guess (0)

    // Create the linear operator for the unsteady system: (I + dt*A)
    // This operator is constant across time steps if dt is constant.
    // Needs dt for its definition.
    UnsteadyOperator linear_operator_unsteady(m_A, dt); // Operator is (I + dt*A)

    // Time stepping loop
    for (int k_time = 0; k_time < nt; ++k_time) {
        double current_time = k_time * dt; // Time at the start of the step (t_k)
        double next_time = (k_time + 1) * dt; // Time at the end of the step (t_{k+1})

        std::cout << "Time Step " << k_time + 1 << "/" << nt << " (t=" << next_time << ")" << std::endl;

        // 1. Assemble the RHS for the unsteady step: U_old + dt * F_base(t_new)
        // Need assembleRHS_unsteady method.
        // This method needs the new time (next_time), dt, and U_old.
        // Assuming assembleRHS_unsteady(RHS, time, dt_step, U_old) exists:
        m_assembler.assembleRHS_unsteady(m_F, next_time, dt, U_old); // m_F becomes the RHS for CG

        // 2. Solve the linear system (I + dt*A) U_new = RHS using CG
        // m_U will hold the new solution U_new. It can be used as the initial guess for CG.
        // The initial guess for CG at each time step can be U_old or the previous CG solution.
        // Fortran used Uk (U_old) as part of the RHS and then solved for U, which became Uk for next step.
        // The solve signature is solve(U_new, RHS, Operator). So U_new is updated in place.
        // Use m_U as the initial guess (e.g., U_old or 0) for the CG solver for U_new.
        // Let's re-zero m_U for each CG solve for simplicity, or pass U_old as the initial guess.
        // Passing U_old might be a better guess. But the CG signature updates U in place.
        // Let's make a copy for the initial guess and pass m_U to be updated.
        Eigen::VectorXd cg_initial_guess = U_old; // Use U_old as guess for U_new

        std::cout << "Solving (I+dt*A)U_new=RHS using CG..." << std::endl;
        // solve(U_new, RHS, Operator, tolerance, max_iters)
        m_cg_solver.solve(m_U, m_F, linear_operator_unsteady, 1e-4, 10000); // Use appropriate tolerance for unsteady

        // 3. Update U_old for the next time step
        U_old = m_U; // The newly computed U becomes U_old for the next step

        // Optional: Save results at specific time steps if needed
        // if ((k_time + 1) % save_interval == 0) { ... }
    }

    // 4. Save the final solution after all time steps
    saveResults("VecteurU_unsteady_final", "VecteurU_b_unsteady_final"); // Use specific filenames for unsteady

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Sequential unsteady solver finished in " << elapsed.count() << " seconds." << std::endl;
}

// Helper method to save the solution vector to files
void SequentialHeatSolver::saveResults(const std::string& filename_U, const std::string& filename_Ub) const {
    // TODO: Implement saving logic.
    // Save m_U to filename_U (one value per line).
    // Save coordinates (j*dx, i*dy) and value U(k) to filename_Ub.
    // Use m_grid.indexToCoords for mapping.

    std::ofstream file_u(filename_U);
    std::ofstream file_ub(filename_Ub);

    if (!file_u.is_open() || !file_ub.is_open()) {
        std::cerr << "Error: Could not open output files " << filename_U << " or " << filename_Ub << " for writing." << std::endl;
        // Handle error (e.g., exit)
        return;
    }

    file_u << std::fixed << std::setprecision(10); // Format output precision
    file_ub << std::fixed << std::setprecision(10);

    int n = m_grid.getNumNodes();
    double dx = m_grid.getDx();
    double dy = m_grid.getDy();

    // Loop through global indices (0-based k)
    for (int k = 0; k < n; ++k) {
        file_u << m_U(k) << std::endl; // Save U value

        int i, j; // 1-based grid coords
        m_grid.indexToCoords(k, i, j);
        file_ub << j * dx << " " << i * dy << " " << m_U(k) << std::endl; // Save coords and value
    }

    file_u.close();
    file_ub.close();
    std::cout << "Results saved to " << filename_U << " and " << filename_Ub << std::endl;
}

// Assuming SequentialSystemAssembler::assembleRHS_steady and assembleRHS_unsteady are implemented
// as discussed in the comments of SequentialSystem.cpp.
// If not, the current assembleRHS would need to handle both cases based on U_old being null or not.
```

---

**File: `include/ParallelHeatSolver.hpp`**

```cpp
#ifndef PARALLEL_HEAT_SOLVER_HPP
#define PARALLEL_HEAT_SOLVER_HPP

#include "GridGeometry.hpp"
#include "PhysicsParameters.hpp"
#include "EquationFunctions.hpp"
#include "ParallelSystem.hpp" // Parallel system setup and mat-vec product
#include "ParallelCG.hpp"     // Parallel CG solver
#include <Eigen/Dense>        // For local and global vectors
#include <mpi.h>              // MPI headers
#include <string>
#include <vector>
#include <memory>             // For std::unique_ptr

// Class to orchestrate the parallel solving process (steady or unsteady)
class ParallelHeatSolver {
private:
    MPI_Comm m_comm;         // MPI Communicator
    int m_rank;              // MPI Rank
    int m_size;              // MPI Size

    // Problem configuration (read and broadcast)
    GridGeometry m_grid;
    PhysicsParameters m_physics;
    EquationFunctions m_eq_funcs; // May need to handle unsteady/steady f separately if not using dynamic functions

    // Parallel system components
    ParallelSystem m_parallel_system;
    ParallelConjugateGradientSolver m_parallel_cg_solver;

    // Data for the parallel system and solution (managed as global vectors on each process)
    // While technically only local parts are needed for computation, CG might require
    // access to the full vector structure or specific ghost exchanges.
    // Let's follow the Fortran lead and manage full global vectors U and F (or DIR)
    // on each process where necessary, relying on methods like parallelMatrixVectorProduct
    // to use only the relevant local/ghost parts and handle communication.
    // Note: For large problems, storing full global vectors on every process is not scalable.
    // A true distributed vector class would be needed (e.g., from PETSc, Trilinos).
    // Sticking to Eigen + MPI requires careful management of what data is truly needed globally vs locally.

    Eigen::VectorXd m_U_global; // Solution vector (full global vector)
    Eigen::VectorXd m_F_global; // RHS vector (full global vector)
    Eigen::VectorXd m_U_old_global; // For unsteady case (full global vector of previous step)

public:
    // Constructor handles MPI setup and reading/broadcasting parameters
    ParallelHeatSolver(MPI_Comm comm, const std::string& data_filename);

    // Virtual destructor to handle MPI_Finalize (if MPI_Init was called here)
    // Note: Standard practice is to call MPI_Init/Finalize in main.
    // If called in main, destructor is not needed for MPI_Finalize.
    // Let's assume MPI_Init/Finalize are in main.
    // ~ParallelHeatSolver(); // No destructor needed for MPI_Finalize if it's in main

    // Method to run the steady-state parallel solver
    void runSteady();

    // Method to run the unsteady parallel solver
    // nt: number of time steps
    // dt: time step size
    void runUnsteady(int nt, double dt);

private:
    // Helper method to read parameters and broadcast them from rank 0
    // Returns a tuple of (Nx, Ny, Lx, Ly, D)
    std::tuple<int, int, double, double, double> readAndBroadcastParameters(const std::string& filename);

    // Helper method to save the solution vector to files (parallel version)
    // Can save locally on each process or gather on rank 0 and save.
    void parallelSaveResults(const Eigen::VectorXd& U_global, const std::string& filename_U_prefix, const std::string& filename_Ub_prefix) const;

    // Helper method to get the local part of a global vector
    Eigen::VectorXd getLocalVectorPart(const Eigen::VectorXd& global_vec) const;
};

#endif // PARALLEL_HEAT_SOLVER_HPP
```

---

**File: `src/ParallelHeatSolver.cpp`**

```cpp
#include "ParallelHeatSolver.hpp"
#include <fstream>      // For file output
#include <iomanip>      // For formatting output
#include <iostream>     // For console output
#include <chrono>       // For timing
#include <numeric>      // For std::iota
#include <tuple>        // For std::get
#include <limits>       // For numeric_limits

// Forward declaration for read_parameters if not in a separate file
// std::tuple<int, int, double, double, double> read_parameters(const std::string& filename);
// Assuming read_parameters is defined elsewhere and accessible (e.g., in a common Utils.cpp)

// Let's define a minimal read_parameters here for the skeleton if not using Utils
namespace { // Anonymous namespace for local helper
std::tuple<int, int, double, double, double> read_parameters_local(const std::string& filename) {
    std::ifstream file(filename);
    int Nx_val, Ny_val;
    double Lx_val, Ly_val, D_val;
    if (file.is_open()) {
        file >> Nx_val >> Ny_val >> Lx_val >> Ly_val >> D_val;
        file.close();
        return std::make_tuple(Nx_val, Ny_val, Lx_val, Ly_val, D_val);
    } else {
        // Handle error
        return std::make_tuple(0, 0, 0.0, 0.0, 0.0); // Return zeros on error
    }
}
} // namespace

// Constructor handles MPI setup and reading/broadcasting parameters
ParallelHeatSolver::ParallelHeatSolver(MPI_Comm comm, const std::string& data_filename)
    : m_comm(comm),
      // Initialize physics and equation functions with default/placeholder values initially
      m_physics(0.0),
      // Need to use static methods or capture variables for functions
      // Using lambdas capturing Lx, Ly, D (which are read/broadcast) is best.
      // Functions will be set after reading parameters.
      m_eq_funcs(
          [](double x, double y, double t){ return 0.0; }, // Placeholder f
          [](double x, double y){ return 0.0; },          // Placeholder g
          [](double x, double y){ return 0.0; }           // Placeholder h
      ),
      // Initialize ParallelSystem and ParallelCG later, as they need grid/physics/eq_funcs refs
      // Need to initialize m_grid first as ParallelSystem needs a GridGeometry reference
      m_grid(0, 0, 0.0, 0.0), // Placeholder grid
      m_parallel_system(m_grid, m_physics, m_eq_funcs, m_comm), // Placeholder initialization
      m_parallel_cg_solver(m_parallel_system, m_comm) // Placeholder initialization
{
    // Get rank and size
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);

    // Read and broadcast parameters
    auto params = readAndBroadcastParameters(data_filename);
    int Nx = std::get<0>(params);
    int Ny = std::get<1>(params);
    double Lx = std::get<2>(params);
    double Ly = std::get<3>(params);
    double D = std::get<4>(params);

    // Now initialize/reset the actual member objects using the broadcasted parameters
    // Reset GridGeometry
    m_grid = GridGeometry(Nx, Ny, Lx, Ly);
    // Reset PhysicsParameters
    m_physics = PhysicsParameters(D);
    // Reset EquationFunctions with correct functions and captured parameters (if needed)
    m_eq_funcs = EquationFunctions(
        // Capture Lx, Ly, D by value in the lambdas if the static functions need them
        [=](double x, double y, double t){ return EquationFunctions::default_f1_unsteady(x, y, t, Lx, Ly); }, // Default to unsteady source
        [=](double x, double y){ return EquationFunctions::default_g(x, y); },
        [=](double x, double y){ return EquationFunctions::default_h(x, y); }
    );

    // Re-initialize ParallelSystem and ParallelCG with correct references
    // This is tricky with references and default constructors.
    // A better approach might be to use pointers or unique_ptrs and allocate after reading parameters.
    // Or, pass parameters to the ParallelSystem constructor *after* reading.

    // *** REVISED PLAN (AGAIN) ***
    // ParallelHeatSolver Constructor:
    // 1. MPI_Comm_rank/size
    // 2. Read/Broadcast parameters.
    // 3. Initialize m_grid, m_physics, m_eq_funcs using broadcasted params.
    // 4. Pass references to these to the ParallelSystem constructor.
    // 5. Pass ParallelSystem reference and comm to ParallelCG constructor.
    // 6. Allocate global vectors m_U_global, m_F_global, m_U_old_global based on total nodes.

    // Redoing the constructor body based on this plan:

    // Read and broadcast parameters
    // (Already done above)

    // Determine total number of nodes
    int n = m_grid.getNumNodes();

    // Allocate global vectors
    m_U_global.resize(n);
    m_F_global.resize(n);
    m_U_old_global.resize(n);

    // Initial guess for U (often zero globally)
    m_U_global.setZero();
    m_U_old_global.setZero(); // Initial condition for unsteady (t=0)

    // Assemble the local matrix once during setup
    if (m_rank == 0) std::cout << "Assembling local matrix A (Parallel)..." << std::endl;
    m_parallel_system.assembleLocalMatrix(); // Uses the grid, physics refs set above
    if (m_rank == 0) std::cout << "Local matrices assembled." << std::endl;
}

// Helper method to read parameters and broadcast them from rank 0
std::tuple<int, int, double, double, double> ParallelHeatSolver::readAndBroadcastParameters(const std::string& filename) {
    // TODO: Implement reading on rank 0 and broadcasting to all ranks.
    // Use MPI_Bcast for fundamental types.
    // std::tuple is convenient for returning multiple values.

    int Nx_val, Ny_val;
    double Lx_val, Ly_val, D_val;

    if (m_rank == 0) {
        // Read from file only on rank 0
        auto params = read_parameters_local(filename); // Using local helper for example
        Nx_val = std::get<0>(params);
        Ny_val = std::get<1>(params);
        Lx_val = std::get<2>(params);
        Ly_val = std::get<3>(params);
        D_val  = std::get<4>(params);
         if (m_rank == 0) {
              std::cout << "Parameters read from " << filename << ":" << std::endl;
              std::cout << "Nx = " << Nx_val << ", Ny = " << Ny_val << std::endl;
              std::cout << "Lx = " << Lx_val << ", Ly = " << Ly_val << std::endl;
              std::cout << "D  = " << D_val << std::endl;
         }
    }

    // Broadcast parameters from rank 0 to all other ranks
    MPI_Bcast(&Nx_val, 1, MPI_INT, 0, m_comm);
    MPI_Bcast(&Ny_val, 1, MPI_INT, 0, m_comm);
    MPI_Bcast(&Lx_val, 1, MPI_DOUBLE, 0, m_comm);
    MPI_Bcast(&Ly_val, 1, MPI_DOUBLE, 0, m_comm);
    MPI_Bcast(&D_val, 1, MPI_DOUBLE, 0, m_comm);

    // Return the broadcasted parameters
    return std::make_tuple(Nx_val, Ny_val, Lx_val, Ly_val, D_val);
}

// Method to run the steady-state parallel solver
void ParallelHeatSolver::runSteady() {
    if (m_rank == 0) std::cout << "Running parallel steady-state solver..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Assemble the global RHS vector F for the steady case
    // Each process assembles its local part, then these can conceptually form the global F.
    // The CG solver needs access to the global F (or its local part + info).
    // Let's assemble the full F_global vector on each process for simplicity,
    // although for true scalability, F would also be distributed.
    // Alternative: Assemble only F_local and pass F_local to CG, then CG needs access to F_local from all processes.
    // Let's stick to assembling F_global on each process for now, similar to U_global and DIR_global.
    // This matches the Fortran pattern where F is dimensioned N even in parallel modules.

    if (m_rank == 0) std::cout << "Assembling RHS vector F (Steady, globally replicated)..." << std::endl;
    // Need assembleLocalRHS_steady method and then maybe gather/allgather F_local into F_global?
    // Or, if F is conceptually global on each process, assemble F_global directly.
    // Let's call assembleLocalRHS_steady and then copy the local part into F_global.
    // This implies F_global is allocated but parts are zero except the local one.
    // This is inefficient. A better way is to pass F_global to assembleLocalRHS and have it write only the local part.
    // *** REVISED PLAN for Vector Management ***
    // - ParallelSystem works with local vectors (m_local_n size).
    // - ParallelCG also works with local vectors (m_local_n size) for Gr, DIR, V etc.
    // - The *solution* vector U and the *RHS* vector F are conceptually global, but maybe only stored globally on rank 0 at the end.
    // - For the parallel Matrix-Vector product `y_local = A_local * x_global`, `x_global` IS needed globally (or needs careful ghost exchange).
    // - So, the main solver class (ParallelHeatSolver) will hold the *full global* vectors m_U_global, m_F_global, m_U_old_global.
    // - When calling `assembleLocalRHS_steady/unsteady`, pass m_F_global and let the method write to the correct segment.
    // - When calling `parallelMatrixVectorProduct`, pass the relevant full vector (like m_U_global or DIR_global).
    // - When calling `parallelCG.solve`, pass m_U_global and m_F_global. CG operates internally with local parts and updates m_U_global's local part.

    // Re-implementing runSteady based on this vector management plan:

    if (m_rank == 0) std::cout << "Assembling RHS vector F (Steady)..." << std::endl;
    // Call the assembler to fill the local part of m_F_global
    m_parallel_system.assembleLocalRHS_steady(m_F_global, 0.0); // time=0.0 for steady


    // 2. Solve the linear system AU = F using Parallel CG
    // Pass m_U_global (initial guess), m_F_global (RHS), and the ParallelSystem object (provides mat-vec).
    // The CG solver will handle accessing local segments and MPI communication.
    if (m_rank == 0) std::cout << "Solving AU=F using Parallel CG..." << std::endl;
    m_parallel_cg_solver.solve(m_U_global, m_F_global, 1e-4, 10000); // Use appropriate tolerance for parallel steady

    // 3. Save the final solution (Parallel Save)
    if (m_rank == 0) std::cout << "Saving final results (Parallel)..." << std::endl;
    parallelSaveResults(m_U_global, "VecteurU_steady_para", "VecteurU_b_steady_para"); // Use specific filenames

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
     if (m_rank == 0) std::cout << "Parallel steady-state solver finished in " << elapsed.count() << " seconds." << std::endl;
}


// Method to run the unsteady parallel solver
// nt: number of time steps
// dt: time step size
void ParallelHeatSolver::runUnsteady(int nt, double dt) {
    if (m_rank == 0) std::cout << "Running parallel unsteady solver for " << nt << " steps with dt=" << dt << "..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // m_U_old_global is the initial condition (already zeroed in constructor)

    // Time stepping loop
    for (int k_time = 0; k_time < nt; ++k_time) {
        double current_time = k_time * dt; // Time at the start of the step (t_k)
        double next_time = (k_time + 1) * dt; // Time at the end of the step (t_{k+1})

         if (m_rank == 0) {
              std::cout << "Time Step " << k_time + 1 << "/" << nt << " (t=" << next_time << ")" << std::endl;
         }

        // 1. Assemble the RHS for the unsteady step: U_old + dt * F_base(t_new)
        // Call the assembler to fill the local part of m_F_global.
        // Needs assembleLocalRHS_unsteady(RHS_global, time, dt_step, U_old_global).
        // This fills the local segment of RHS_global.
        m_parallel_system.assembleLocalRHS_unsteady(m_F_global, next_time, dt, m_U_old_global); // m_F_global now holds RHS for CG

        // 2. Solve the linear system (I + dt*A) U_new = RHS using Parallel CG
        // The operator is implicitly handled by the ParallelSystem::parallelMatrixVectorProduct method
        // which is called by ParallelCG. solve(U_global, RHS_global, tolerance, max_iters).
        // The solve method will compute U_new and store it in m_U_global.
        // m_U_global initially holds the initial guess for U_new (can be U_old, or zero).
        // Let's use m_U_old_global as the initial guess for m_U_global for efficiency.
        // Need to copy U_old to U first.
        m_U_global = m_U_old_global; // U_global is now the initial guess for the CG solve

        if (m_rank == 0) std::cout << "Solving (I+dt*A)U_new=RHS using Parallel CG..." << std::endl;
        // Call parallel CG solver. It operates on m_U_global and m_F_global internally.
        m_parallel_cg_solver.solve(m_U_global, m_F_global, 1e-4, 10000); // Use appropriate tolerance for parallel unsteady

        // 3. Update U_old_global for the next time step
        m_U_old_global = m_U_global; // The newly computed U becomes U_old for the next step

        // Optional: Parallel Save results at specific time steps if needed
        // if ((k_time + 1) % save_interval == 0) { ... }
    }

    // 4. Save the final solution after all time steps (Parallel Save)
     if (m_rank == 0) std::cout << "Saving final results (Parallel)..." << std::endl;
    parallelSaveResults(m_U_global, "VecteurU_unsteady_para_final", "VecteurU_b_unsteady_para_final"); // Use specific filenames

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
     if (m_rank == 0) std::cout << "Parallel unsteady solver finished in " << elapsed.count() << " seconds." << std::endl;
}

// Helper method to save the solution vector to files (parallel version)
// Can save locally on each process or gather on rank 0 and save.
void ParallelHeatSolver::parallelSaveResults(const Eigen::VectorXd& U_global, const std::string& filename_U_prefix, const std::string& filename_Ub_prefix) const {
    // TODO: Implement parallel saving.
    // Option A (Simpler, less scalable): Gather U_global onto rank 0, then rank 0 saves the full vector.
    // Option B (More scalable): Each process saves its local portion to a distinct file (e.g., filename.rank).
    // Option C (Most scalable): Use parallel I/O libraries (e.g., HDF5 with MPI-IO, Parallel NetCDF).

    // Let's implement Option A (Gather on rank 0) for simplicity in the skeleton.
    // This requires U_global to be valid (likely the final result after parallel CG).

    if (m_rank == 0) {
        // Only rank 0 opens and writes the files
        std::ofstream file_u(filename_U_prefix);
        std::ofstream file_ub(filename_Ub_prefix);

        if (!file_u.is_open() || !file_ub.is_open()) {
            std::cerr << "Error: Rank 0 could not open output files for writing." << std::endl;
            // Handle error
            return;
        }

        file_u << std::fixed << std::setprecision(10); // Format output precision
        file_ub << std::fixed << std::setprecision(10);

        int n = m_grid.getNumNodes();
        double dx = m_grid.getDx();
        double dy = m_grid.getDy();

        // Assuming U_global has been successfully assembled/updated across processes
        // and rank 0 has the complete vector (e.g., if CG result is broadcast back to U_global).
        // If CG only updates local part of U_global, rank 0 might need to gather it first.
        // Let's assume the ParallelCG solve updates U_global on all processes for simplicity here.
        // If CG only updates local part, add MPI_Gather call here before the loop.
        // MPI_Gather(U_global.segment(m_parallel_system.getLocalStartIndex(), m_parallel_system.getLocalNumNodes()).data(),
        //            m_parallel_system.getLocalNumNodes(), MPI_DOUBLE,
        //            U_global.data(), m_parallel_system.getLocalNumNodes(), MPI_DOUBLE, 0, m_comm);
        // The ParallelCG::solve method should ideally update the U_global vector passed to it.

        // Loop through global indices (0-based k) on rank 0
        for (int k = 0; k < n; ++k) {
            file_u << U_global(k) << std::endl; // Save U value

            int i, j; // 1-based grid coords
            m_grid.indexToCoords(k, i, j);
            file_ub << j * dx << " " << i * dy << " " << U_global(k) << std::endl; // Save coords and value
        }

        file_u.close();
        file_ub.close();
        std::cout << "Rank 0 saved results to " << filename_U_prefix << " and " << filename_Ub_prefix << std::endl;
    }
    // All processes must reach this point together before potentially exiting MPI
    MPI_Barrier(m_comm);
}

// Helper method to get the local part of a global vector (optional, can use Eigen::segment directly)
Eigen::VectorXd ParallelHeatSolver::getLocalVectorPart(const Eigen::VectorXd& global_vec) const {
     int i1 = m_parallel_system.getLocalStartIndex();
     int local_n = m_parallel_system.getLocalNumNodes();
     return global_vec.segment(i1, local_n);
}

// Assuming ParallelSystem::assembleLocalRHS_steady and assembleLocalRHS_unsteady are implemented
// as discussed in the comments of ParallelSystem.cpp.
// Also assumes ParallelSystem::parallelMatrixVectorProduct is implemented.
// Assumes ParallelCG::solve updates the U_global vector passed to it.
```

---

**File: `src/SequentialHeatSolver.cpp` (Updated function signatures)**

*(Note: The previous content for this file is kept, but remember the internal logic needs to match the updated signatures for `assembleRHS_steady` and `assembleRHS_unsteady` if you modify the base class/assembler as discussed in the comments.)*

---

**File: `src/main_sequential.cpp`**

```cpp
#include "SequentialHeatSolver.hpp"
#include "EquationFunctions.hpp" // Include if using static default functions
#include <iostream>
#include <tuple> // For std::tuple

// Helper function to read parameters (can be moved to a common Utils file)
std::tuple<int, int, double, double, double> read_parameters(const std::string& filename) {
    std::ifstream file(filename);
    int Nx_val, Ny_val;
    double Lx_val, Ly_val, D_val;
    if (file.is_open()) {
        file >> Nx_val >> Ny_val >> Lx_val >> Ly_val >> D_val;
        file.close();
        std::cout << "Parameters read from " << filename << ":" << std::endl;
        std::cout << "Nx = " << Nx_val << ", Ny = " << Ny_val << std::endl;
        std::cout << "Lx = " << Lx_val << ", Ly = " << Ly_val << std::endl;
        std::cout << "D  = " << D_val << std::endl;
        return std::make_tuple(Nx_val, Ny_val, Lx_val, Ly_val, D_val);
    } else {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        // Handle error appropriately
        exit(EXIT_FAILURE);
    }
}


int main() {
    // Read parameters
    auto params = read_parameters("./data");
    int Nx = std::get<0>(params);
    int Ny = std::get<1>(params);
    double Lx = std::get<2>(params);
    double Ly = std::get<3>(params);
    double D = std::get<4>(params);

    // Define the equation functions (using static defaults from EquationFunctions)
    // Capture Lx, Ly, D by value if default functions need them
    auto f_steady = [=](double x, double y, double t){ return EquationFunctions::default_f1_stationary(x, y, t); };
    auto f_unsteady = [=](double x, double y, double t){ return EquationFunctions::default_f1_unsteady(x, y, t, Lx, Ly); }; // Pass Lx, Ly
    auto g_bc = [=](double x, double y){ return EquationFunctions::default_g(x, y); };
    auto h_bc = [=](double x, double y){ return EquationFunctions::default_h(x, y); };

    // Create the sequential solver object
    // Pass the appropriate source term function depending on whether solving steady or unsteady
    SequentialHeatSolver solver_steady(Nx, Ny, Lx, Ly, D, f_steady, g_bc, h_bc);
    // For unsteady, you'd instantiate potentially with the unsteady source function
    SequentialHeatSolver solver_unsteady(Nx, Ny, Lx, Ly, D, f_unsteady, g_bc, h_bc);


    // --- Choose which solver to run ---

    // Run steady-state solver
    std::cout << "\n--- Running Steady Sequential Solver ---" << std::endl;
    solver_steady.runSteady();

    // Run unsteady solver (example parameters)
    // std::cout << "\n--- Running Unsteady Sequential Solver ---" << std::endl;
    // int nt = 100; // Number of time steps
    // double dt = 0.1; // Time step size
    // solver_unsteady.runUnsteady(nt, dt);


    std::cout << "\nSequential Main program finished." << std::endl;

    return 0;
}
```

---

**File: `src/main_parallel.cpp`**

```cpp
#include "ParallelHeatSolver.hpp"
#include "EquationFunctions.hpp" // Include if using static default functions
#include <iostream>
#include <tuple> // For std::tuple
#include <mpi.h> // For MPI

// Helper function to read parameters (can be moved to a common Utils file)
// Note: In the parallel case, rank 0 reads, others wait for broadcast.
// This function is just for rank 0 reading before broadcast.
std::tuple<int, int, double, double, double> read_parameters_parallel_rank0(const std::string& filename) {
    std::ifstream file(filename);
    int Nx_val, Ny_val;
    double Lx_val, Ly_val, D_val;
    if (file.is_open()) {
        file >> Nx_val >> Ny_val >> Lx_val >> Ly_val >> D_val;
        file.close();
        return std::make_tuple(Nx_val, Ny_val, Lx_val, Ly_val, D_val);
    } else {
        std::cerr << "Error: Rank 0 unable to open file " << filename << std::endl;
        // Handle error (exit or throw)
        return std::make_tuple(0, 0, 0.0, 0.0, 0.0); // Return zeros on error
    }
}


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    MPI_Comm world_comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(world_comm, &rank);
    MPI_Comm_size(world_comm, &size);

    if (rank == 0) {
        std::cout << "MPI Initialized with " << size << " processes." << std::endl;
    }

    // Note: Parameters are read and broadcast inside the ParallelHeatSolver constructor now.

    // Define the equation functions (using static defaults from EquationFunctions)
    // Need to create these AFTER parameters are read and broadcasted,
    // because the lambdas might need to capture Lx, Ly, D.
    // The ParallelHeatSolver constructor will handle this internally after broadcasting.
    // Just need to pass the filename to the constructor.

    // Create the parallel solver object
    // Pass the MPI communicator and the data filename.
    // The constructor will read params, broadcast, initialize members, and assemble A_local.
    ParallelHeatSolver solver(world_comm, "./data");


    // --- Choose which solver to run ---

    // Run steady-state parallel solver
    std::cout << std::endl; // Print empty line for clarity on all ranks
    solver.runSteady();

    // Run unsteady parallel solver (example parameters)
    // std::cout << std::endl; // Print empty line for clarity on all ranks
    // int nt = 10; // Number of time steps
    // double dt = 0.1; // Time step size
    // solver.runUnsteady(nt, dt);


    if (rank == 0) {
        std::cout << "\nParallel Main program finished." << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
```

---

**File: `Makefile`**

```makefile
# ==============================================================================
# Makefile for Heat Equation Solver (Sequential and Parallel Versions)
# ==============================================================================

# Define the C++ compiler (use mpicxx for parallel versions)
CXX = mpicxx
# Define compilation flags
CXXFLAGS = -std=c++11 -Wall -Wextra -g # Standard C++11, warnings, debug info
# Include Eigen headers (adjust path if necessary)
# Example: If Eigen is in /usr/local/include/eigen3
# EIGEN_INCLUDE = -I/usr/local/include/eigen3
# Assuming Eigen is in a standard include path or you adjust EIGEN_INCLUDE
EIGEN_INCLUDE =

# Define linker flags (often same as CXXFLAGS)
LDFLAGS = $(CXXFLAGS)

# Default number of MPI processes for parallel runs
NBR_PROC = 4 # Adjust this number as needed

# MPI run command
MPI_CMD = mpirun -np

# Source and Include directories
SRC_DIR = src
INCLUDE_DIR = include

# List source files for the common utilities and classes
COMMON_SRC = \
	$(SRC_DIR)/GridGeometry.cpp \
	$(SRC_DIR)/EquationFunctions.cpp \
	$(SRC_DIR)/SequentialSystem.cpp \
	$(SRC_DIR)/ParallelSystem.cpp \
	$(SRC_DIR)/SequentialCG.cpp \
	$(SRC_DIR)/ParallelCG.cpp \
	$(SRC_DIR)/MatrixOperator.cpp \
	$(SRC_DIR)/UnsteadyOperator.cpp \
	# Add other common utility source files if created (e.g., Utils.cpp)

# List source files for the main programs
MAIN_SEQ_SRC = $(SRC_DIR)/main_sequential.cpp
MAIN_PAR_SRC = $(SRC_DIR)/main_parallel.cpp

# Object files for common components
COMMON_OBJ = $(COMMON_SRC:.cpp=.o)

# Object files for main programs
MAIN_SEQ_OBJ = $(MAIN_SEQ_SRC:.cpp=.o)
MAIN_PAR_OBJ = $(MAIN_PAR_SRC:.cpp=.o)

# Executable names
SEQUENTIAL_EXE = heat_solver_sequential
PARALLEL_EXE = heat_solver_parallel

# ==============================================================================
# Build Targets
# ==============================================================================

# Default target: Build both executables
all: $(SEQUENTIAL_EXE) $(PARALLEL_EXE)
	@echo "All targets built successfully."

# Rule to link the sequential executable
$(SEQUENTIAL_EXE): $(MAIN_SEQ_OBJ) $(COMMON_OBJ)
	@echo "Linking sequential solver: $@"
	$(CXX) $(LDFLAGS) $(EIGEN_INCLUDE) -o $@ $^

# Rule to link the parallel executable
$(PARALLEL_EXE): $(MAIN_PAR_OBJ) $(COMMON_OBJ)
	@echo "Linking parallel solver: $@"
	$(CXX) $(LDFLAGS) $(EIGEN_INCLUDE) -o $@ $^

# Generic rule to compile .cpp files to .o files
# Use -I$(INCLUDE_DIR) to find headers
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(EIGEN_INCLUDE) -I$(INCLUDE_DIR) -c $< -o $@

# ==============================================================================
# Dependencies
# Make will automatically handle most dependencies using included header files
# if the compiler supports it (like g++/mpicxx).
# Explicit dependencies are usually not needed for every single header include.
# However, you might add key ones for clarity or robustness if needed.
# Example:
# $(SRC_DIR)/SequentialSystem.o: $(INCLUDE_DIR)/SequentialSystem.hpp $(INCLUDE_DIR)/GridGeometry.hpp $(INCLUDE_DIR)/PhysicsParameters.hpp $(INCLUDE_DIR)/EquationFunctions.hpp $(INCLUDE_DIR)/SystemBase.hpp
# ... and so on for other .o files on the headers they directly include.
# Let's omit explicit dependencies for simplicity in this skeleton,
# relying on the compiler's dependency generation.
# ==============================================================================


# ==============================================================================
# Run Targets
# ==============================================================================

# Default run target (can be adjusted)
run: run_parallel # Defaulting to parallel unsteady, but this skeleton is just steady

# Run the sequential program (Steady case in main_sequential)
run_sequential: $(SEQUENTIAL_EXE)
	@echo "Running sequential solver..."
	./$(SEQUENTIAL_EXE)

# Run the parallel program (Steady case in main_parallel)
run_parallel: $(PARALLEL_EXE)
	@echo "Running parallel solver with $(NBR_PROC) processes..."
	$(MPI_CMD) $(NBR_PROC) --allow-run-as-root ./$(PARALLEL_EXE)

# Specific run targets for steady/unsteady versions (if implemented in main)
# You'd uncomment and adjust these if your main files have flags/logic
# to switch between steady/unsteady modes. The current skeleton mains
# only run the steady case by default.

# run_sequential_steady: $(SEQUENTIAL_EXE)
# 	@echo "Running sequential steady solver..."
# 	./$(SEQUENTIAL_EXE) steady

# run_sequential_unsteady: $(SEQUENTIAL_EXE)
# 	@echo "Running sequential unsteady solver..."
# 	./$(SEQUENTIAL_EXE) unsteady # Need to add command line arg handling in main

# run_parallel_steady: $(PARALLEL_EXE)
# 	@echo "Running parallel steady solver..."
# 	$(MPI_CMD) $(NBR_PROC) --allow-run-as-root ./$(PARALLEL_EXE) steady

# run_parallel_unsteady: $(PARALLEL_EXE)
# 	@echo "Running parallel unsteady solver..."
# 	$(MPI_CMD) $(NBR_PROC) --allow-run-as-root ./$(PARALLEL_EXE) unsteady # Need to add command line arg handling in main


# ==============================================================================
# Clean Targets
# ==============================================================================

# Files to clean
CLEAN_EXEC = $(SEQUENTIAL_EXE) $(PARALLEL_EXE)
CLEAN_OBJ = $(COMMON_OBJ) $(MAIN_SEQ_OBJ) $(MAIN_PAR_OBJ)

# Output files generated by the code (list specific names and patterns)
# Update this list based on what your C++ code actually outputs
CLEAN_OUTPUT_FILES = \
	$(wildcard Mat* VectF* VectU*) \
	VecteurF \
	Matrice_Quadri \
	VecteurU_b \
	VecteurU \
	VecteurM \
	175 # Assuming 175 was an output file

# Clean object files
clean:
	@echo "Cleaning object files..."
	rm -f $(CLEAN_OBJ)

# Full clean: Remove executables and all output files as well
fclean: clean
	@echo "Cleaning executables and output files..."
	rm -f $(CLEAN_EXEC) $(CLEAN_OUTPUT_FILES)

# Rebuild: Clean completely then build all
re: fclean all
	@echo "Rebuild complete."

# ==============================================================================
# Phony Targets
# ==============================================================================

# Declare targets that are not actual files to avoid conflicts
.PHONY: all run clean fclean re \
		run_sequential run_parallel \
		run_sequential_steady run_sequential_unsteady \
		run_parallel_steady run_parallel_unsteady

```

---

**File: `README.md`**

```markdown
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

```

This comprehensive skeleton should provide a clear roadmap for translating your Fortran code into a well-structured C++ OOP project. Remember to break down the work into small, testable steps when filling in the implementations. Good luck!
