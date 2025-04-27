# --- Compiler and Flags ---

# C++ Compiler
CXX = g++
# CXX = clang++ # Uncomment this line to use clang++ instead

# Compiler Flags:
# -std=c++11 : Use the C++11 standard (Eigen requires at least C++03, but C++11 is common)
# -Wall      : Enable most common warnings
# -Wextra    : Enable extra warnings
# -pedantic  : Enable strict ISO C++ compliance warnings
# -g         : Generate debugging information
# -O2        : Optimization level 2 (use -O0 for no optimization, -O3 for more)
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -g -O0

# Linker Flags (usually empty for simple projects)
LDFLAGS =

# Libraries to Link (e.g., -lm for math library if needed, Eigen itself is header-only)
LDLIBS =

# --- Eigen3 Detection ---

# Try to get Eigen3 include path using pkg-config (works if Eigen was installed via package manager correctly)
# The 'shell' function executes the command. '2>/dev/null' suppresses errors if pkg-config fails.
# EIGEN_CFLAGS := $(shell pkg-config --cflags eigen3 2>/dev/null)

# # Check if the user provided EIGEN_INCLUDE_DIR on the command line (e.g., make EIGEN_INCLUDE_DIR=/opt/eigen)
# ifdef EIGEN_INCLUDE_DIR
# 	# If user provided it, use that path directly
# 	USER_EIGEN_FLAG = -I$(EIGEN_INCLUDE_DIR)
# 	CXXFLAGS += $(USER_EIGEN_FLAG)
# 	$(info Using user-provided EIGEN_INCLUDE_DIR: $(EIGEN_INCLUDE_DIR))
# # If user didn't provide it, check if pkg-config found something
# else ifneq ($(EIGEN_CFLAGS),)
# 	# If pkg-config succeeded, add its flags
# 	CXXFLAGS += $(EIGEN_CFLAGS)
#     $(info Using Eigen3 flags from pkg-config: $(EIGEN_CFLAGS))
# # If neither worked, display an error message and stop
# else
#     $(error Cannot find Eigen3 include path automatically using pkg-config. \
# 	        Please specify it manually by running: \
# 	        make EIGEN_INCLUDE_DIR=/path/to/folder/containing/Eigen)
# endif

# --- Project Files ---

# Executable name
TARGET = solver

# Find all .cpp source files in the current directory
SRCS = main.cpp src/GridParameters.cpp \
				src/Vectors.cpp

# Generate corresponding object file names (replace .cpp with .o)
OBJS = $(SRCS:.cpp=.o)


# --- Build Rules ---

# Default target: Build the executable ('all' is a common name)
# This rule depends on the executable file $(TARGET)
all: $(TARGET)

# Rule to link the executable from object files
# $@ is the target name ($(TARGET))
# $^ is the list of all prerequisites ($(OBJS))
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	@echo "Linking complete: $(TARGET)"

# Rule to compile a .cpp source file into a .o object file
# $< is the first prerequisite (the .cpp file)
# $@ is the target name (the .o file)
# -c tells the compiler to compile but not link
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

# --- Cleanup Rule ---
clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) # -f ignores errors if files don't exist

# Target to remove generated files
fclean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(TARGET) *.out # -f ignores errors if files don't exist
	rm -f Mat* VectF* VectU* Vecteur* # Remove specific files if they exist

re: fclean all

run: re
	@echo "Running $(TARGET)..."
	./$(TARGET)


# --- Phony Targets ---

# Declare targets that don't represent actual files
# This prevents 'make' from getting confused if a file named 'all' or 'clean' exists
.PHONY: all clean fclean re run