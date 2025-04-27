#include "../include/Vectors.hpp"
#include <cstddef> // For size_t
#include <vector> // Internal storage
#include <numeric> // For std::inner_product

// Constructor
SequentialVector::SequentialVector(size_t global_size)
{
	data.resize(global_size); // Allocate memory for the vector
}

// Helper function to cast VectorBase to SequentialVector
const SequentialVector& SequentialVector::castToSequential(const VectorBase& base_vector) const
{
	// Attempt to cast the VectorBase reference to SequentialVector reference
	const SequentialVector* seq_vector = dynamic_cast<const SequentialVector*>(&base_vector);
	if (!seq_vector) 
	{
		throw std::runtime_error("Invalid vector type for operation");
	}
	return *seq_vector; // Return the casted reference
}

// --- Implementations of VectorBase methods ---
size_t SequentialVector::getGlobalSize() const
{
	return data.size(); // Return the size of the vector
}

// --- Vector initialization ---
void SequentialVector::ones()
{
	std::fill(data.begin(), data.end(), 1.0); // Fill the vector with 1.0
}

void SequentialVector::zeros()
{
	std::fill(data.begin(), data.end(), 0.0); // Fill the vector with 0.0
}

/***
 * @note []{} is the equivalent of lambda in python
 */
void SequentialVector::random()
{
	std::generate(data.begin(), data.end(), []() { return static_cast<double>(rand()) / RAND_MAX; }); // Fill with random values
}

// --- Vector operations ---
double SequentialVector::dot(const VectorBase& other) const
{
	double dot_product = 0.0;
	// Cast the other vector to SequentialVector for dot product
	const SequentialVector& other_seq = castToSequential(other);
	dot_product = std::inner_product(data.begin(), data.end(), other_seq.data.begin(), 0.0); // the 0.0 is the initial value
	return dot_product;
}

/**
 * @brief Scales the vector by a given factor.
 * it calcules this = this + alpha * y
 * @param alpha The scaling factor.
 * @param y The vector to be added.	
 */
void SequentialVector::axpy(double alpha, const VectorBase& y)
{
	// Cast the other vector to SequentialVector for axpy operation
	const SequentialVector& y_seq = castToSequential(y);
	for (size_t i = 0; i < data.size(); ++i) 
	{
		data[i] += alpha * y_seq.data[i]; // Perform the axpy operation
	}
}

/***
 * @note it calcules this = alpha * this
 */
void SequentialVector::scale(double alpha)
{
	for (size_t i = 0; i < data.size(); ++i) 
	{
		data[i] *= alpha; // Scale the vector
	}
}

void SequentialVector::copy(const VectorBase& other)
{
	// Cast the other vector to SequentialVector for copy operation
	const SequentialVector& other_seq = castToSequential(other);

	if (data.size() != other_seq.data.size()) 
	{
		throw std::runtime_error("Size mismatch during copy");
	}
	data = other_seq.data; // Copy the data
}

double SequentialVector::norm() const
{
	double norm_value = 0.0;

	norm_value = std::sqrt(std::inner_product(data.begin(), data.end(), data.begin(), 0.0)); // L2 norm
	return norm_value;
}