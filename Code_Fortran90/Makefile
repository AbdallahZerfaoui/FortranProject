# Compiler and flags
FC = mpifort
# FFLAGS = -O0 -fcheck=all -g -fbacktrace -ffpe-trap=invalid,zero,overflow
FFLAGS = -g
LDFLAGS = $(FFLAGS)
NBR_PROC = 1

# Source files (order matters for modules!)
SRC = mod_fonctions_instationnaire.f90 \
      Grad_para.f90 \
      mod_remplissage_instationnaire.f90 \
      remplissage_V_para_instationnaire.f90

# Object files (derived from source files)
OBJ = $(SRC:.f90=.o)

# Executable name
EXE = solver

# MPI run command
MPI_CMD = mpirun -np

# Default target
all: $(EXE)

# Rule to link object files into executable
$(EXE): $(OBJ)
	$(FC) $(LDFLAGS) -o $@ $^

# Rule to compile .f90 files to .o files
%.o: %.f90
	$(FC) $(FFLAGS) -c $<

# Run the program
run:
	$(MPI_CMD) $(NBR_PROC) --allow-run-as-root ./$(EXE)

# Clean build artifacts
clean:
	rm -f $(OBJ) $(EXE) *.mod
	rm -f Mat* VectF* VectU*

re: clean all
	@echo "Rebuild complete."


.PHONY: all run clean