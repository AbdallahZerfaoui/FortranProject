# Compiler and flags
FC = mpifort
# FFLAGS = -O0 -fcheck=all -g -fbacktrace -ffpe-trap=invalid,zero,overflow
FFLAGS = -g
LDFLAGS = $(FFLAGS)
NBR_PROC = 4

# Source files (order matters for modules!)
SEQUENIAL_SRC = mod_fonctions_instationnaire.f90 \
	  Grad.f90 \
	  remplissage_V.f90

STEADY_PARALLEL_SRC = mod_fonctions_instationnaire.f90 \
	  Grad_para.f90 \
	  mod_remplissage.f90 \
	  remplissage_V_para_stationnaire.f90

UNSTEADY_PARALLEL_SRC = mod_fonctions_instationnaire.f90 \
      Grad_para.f90 \
      mod_remplissage_instationnaire.f90 \
      remplissage_V_para_instationnaire.f90

# Object files (derived from source files)
SEQUENIAL_OBJ = $(SEQUENIAL_SRC:.f90=.o)
STEADY_PARALLEL_OBJ = $(STEADY_PARALLEL_SRC:.f90=.o)
UNSTEADY_PARALLEL_OBJ = $(UNSTEADY_PARALLEL_SRC:.f90=.o)

# Executable name
SEQUENIAL_EXE = sequential_solver
STEADY_PARALLEL_EXE = steady_para_solver
UNSTEADY_PARALLEL_EXE = unsteady_para_solver

# MPI run command
MPI_CMD = mpirun -np

# Default target
all: $(SEQUENIAL_EXE) $(STEADY_PARALLEL_EXE) $(UNSTEADY_PARALLEL_EXE)
	@echo "All targets built successfully."

# Rule to link object files into executable
$(SEQUENIAL_EXE): $(SEQUENIAL_OBJ)
	$(FC) $(LDFLAGS) -o $@ $^

$(STEADY_PARALLEL_EXE): $(STEADY_PARALLEL_OBJ)
	$(FC) $(LDFLAGS) -o $@ $^

$(UNSTEADY_PARALLEL_EXE): $(UNSTEADY_PARALLEL_OBJ)
	$(FC) $(LDFLAGS) -o $@ $^

# Rule to compile .f90 files to .o files
%.o: %.f90
	$(FC) $(FFLAGS) -c $<

# Run the program
run_sequential: $(SEQUENIAL_EXE)
	@echo "Running sequential version..."
	./$(SEQUENIAL_EXE)

run_steady_parallel: $(STEADY_PARALLEL_EXE)
	@echo "Running steady parallel version..."
	$(MPI_CMD) $(NBR_PROC) --allow-run-as-root ./$(STEADY_PARALLEL_EXE)

run_unsteady_parallel: $(UNSTEADY_PARALLEL_EXE)
	@echo "Running unsteady parallel version..."
	$(MPI_CMD) $(NBR_PROC) --allow-run-as-root ./$(UNSTEADY_PARALLEL_EXE)

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	rm -f $(SEQUENIAL_OBJ) $(UNSTEADY_PARALLEL_OBJ) $(STEADY_PARALLEL_OBJ) *.mod

fclean: clean
	rm -f $(SEQUENIAL_EXE) $(UNSTEADY_PARALLEL_EXE) $(STEADY_PARALLEL_EXE)
	rm -f Mat* VectF* VectU* Vecteur*

re: fclean all
	@echo "Rebuild complete."


.PHONY: all run clean fclean re run_sequential