#ifndef GRIDPARAMETERS_HPP
# define GRIDPARAMETERS_HPP

# include <iostream> // Common include, adjust as needed
# include <string>   // Common include, adjust as needed

// Add other necessary includes here

class GridParameters
{
private:
	int _n;
	int _Nx, _Ny;
	double _Lx, _Ly;
	double _D;
	double _dx, _dy;

public:
	// Canonical Form
	GridParameters(int Nx, int Ny, double Lx, double Ly, double D);                            // Default constructor
	GridParameters(const GridParameters& other); // Copy constructor
	GridParameters& operator=(const GridParameters& other); // Copy assignment operator
	~GridParameters();                           // Destructor	// Add other member functions here (declared above or add more manually)

	void globalIndexToGrid(int k, int& i, int& j) const;
	int gridIndexToGlobal(int i, int j) const;
	std::tuple<int, int, double, double, double> loadConfigData() const;
};


#endif /* GRIDPARAMETERS_HPP */
