#pragma once // Or include guard

#include <cstddef> // For size_t
#include <vector> // Internal storage
#include <numeric> // For std::inner_product
#include "GridParameters.hpp" // Needs GridParameters for Nx (halo size)

// Forward declaration for methods that take/return other vectors
// Needed because VectorBase methods refer to VectorBase objects
class VectorBase; //TODO: I need to understand this

class VectorBase
{
	public:
		// Virtual destructor is crucial for polymorphism
		virtual ~VectorBase() = default;

		// Pure virtual methods defining the vector interface
		virtual size_t getGlobalSize() const = 0; // Total elements across all processes
		// virtual size_t getLocalSize() const = 0;  // Elements on this process
		// virtual double* getLocalData() = 0;       // Write access to local data
		// virtual const double* getLocalData() const = 0; // Read access to local data

		// Standard vector operations - implementations must handle sequential/parallel
		virtual double dot(const VectorBase& other) const = 0;          // Global dot product
		virtual void axpy(double alpha, const VectorBase& y) = 0;       // this = this + alpha * y (local operation)
		virtual void scale(double alpha) = 0;                           // this = alpha * this (local operation)
		virtual void copy(const VectorBase& other) = 0;                 // this = other (local operation)
		virtual double norm() const = 0;                                // Global L2 norm

		// Optional: Element-wise operations might be added if needed frequently
		// virtual void add(const VectorBase& other) = 0; // this = this + other
		// virtual void subtract(const VectorBase& other) = 0; // this = this - other
};


class SequentialVector : public VectorBase 
{
	private:
		std::vector<double> data;

	public:
		// Constructor: Allocates memory for the vector
		SequentialVector(size_t global_size);

		// --- Implementations of VectorBase methods ---
		size_t getGlobalSize() const override;
		// size_t getLocalSize() const override; // Same as global_size for sequential
		// double* getLocalData() override;
		// const double* getLocalData() const override;

		//---- Vector initialization ----
		void ones(); // Initializes all elements to 1.0
		void zeros(); // Initializes all elements to 0.0
		void random(); // Initializes all elements to random values
		// void setValue(double value); // Sets all elements to a specific value
		// void setValuesFromFile(const std::string& filename); // Sets values from a file


		//---- Vector operations ----
		double dot(const VectorBase& other) const override; // Computes standard dot product
		void axpy(double alpha, const VectorBase& y) override; // Performs axpy on std::vector
		void scale(double alpha) override; // Scales std::vector
		void copy(const VectorBase& other) override; // Copies std::vector
		double norm() const override; // Computes standard L2 norm

		// No specific destructor needed unless managing other resources; std::vector handles its memory

	private:
		// Helper to safely cast VectorBase& to SequentialVector& (used internally in methods)
		const SequentialVector& castToSequential(const VectorBase& other) const;
};

// class DistributedVector : public VectorBase {
// 	private:
// 		// Stores the elements of the vector owned by this process
// 		std::vector<double> local_data;
	
// 		// Buffer containing local_data plus halo regions from neighbors
// 		// Size is local_size + 2*Nx
// 		std::vector<double> global_data_with_halos;
	
// 		size_t global_n;         // Total number of elements across all processes
// 		int local_start_idx;     // 0-based global index of the first local element
// 		int local_end_idx;       // 0-based global index of the last local element
// 		int Nx;                  // Number of grid points in x (determines halo size)
	
// 		const MpiHandler* mpi_; // Pointer to the MPI handler instance (not owned)
	
// 	public:
// 		// Default constructor
// 		DistributedVector();
	
// 		// Allocation method: Initializes and allocates memory based on global size, grid, and MPI rank
// 		void allocate(size_t global_size, const GridParameters& grid, const MpiHandler& mpi);
	
// 		// Destructor: Handles memory cleanup via std::vector
// 		~DistributedVector() override = default;
	
// 		// --- Implementations of VectorBase methods ---
// 		size_t getGlobalSize() const override;
// 		size_t getLocalSize() const override; // Returns local_data.size()
// 		double* getLocalData() override;       // Returns local_data.data()
// 		const double* getLocalData() const override; // Returns local_data.data()
	
// 		// Computes local dot product, then performs MPI_Allreduce (sum)
// 		double dot(const VectorBase& other) const override;
	
// 		// Performs axpy operation only on local_data
// 		void axpy(double alpha, const VectorBase& y) override;
	
// 		// Scales only local_data
// 		void scale(double alpha) override;
	
// 		// Copies data only into local_data
// 		void copy(const VectorBase& other) override;
	
// 		// Computes local squared norm sum, then performs MPI_Allreduce (sum) and sqrt
// 		double norm() const override;
	
// 		// --- Parallel-specific methods ---
	
// 		// Performs MPI halo exchange to populate global_data_with_halos
// 		void updateHalos() const;
	
// 		// Provides access to the halo buffer for matrix multiplication
// 		const double* getGlobalDataWithHalos() const;
	
// 		// Returns the offset in global_data_with_halos where local data starts (typically Nx)
// 		size_t getHaloOffset() const;
	
// 		// Get the global index range for this process
// 		int getLocalStartIdx() const { return local_start_idx; }
// 		int getLocalEndIdx() const { return local_end_idx; }
	
	
// 	private:
// 		// Helper to safely cast VectorBase& to DistributedVector& (used internally)
// 		const DistributedVector& castToDistributed(const VectorBase& other) const;
	
// 		// Helper to get size information from the GridParameters (maybe combine with allocate?)
// 		// void setupSizeInfo(const GridParameters& grid);
// 	};