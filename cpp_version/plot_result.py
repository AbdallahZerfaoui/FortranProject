import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'
import numpy as np
import sys # To exit gracefully on error

def plot_data_from_file(filepath):
    """
    Reads data (x, y, value) from a file and creates a 3D scatter plot.

    Args:
        filepath (str): The path to the input data file.
    """
    x_coords = []
    y_coords = []
    values = []

    print(f"Attempting to read data from: {filepath}")

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip() # Remove leading/trailing whitespace
                if not line or line.startswith('#'): # Skip empty lines or comments
                    continue

                parts = line.split() # Split by any whitespace
                if len(parts) == 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        val = float(parts[2])
                        x_coords.append(x)
                        y_coords.append(y)
                        values.append(val)
                    except ValueError:
                        print(f"Warning: Could not parse numbers on line {i+1}. Skipping: '{line}'")
                else:
                    print(f"Warning: Line {i+1} does not contain exactly 3 columns. Skipping: '{line}'")

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1) # Exit the script
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        sys.exit(1)

    if not x_coords: # Check if any data was actually read
        print("Error: No valid data points were read from the file.")
        sys.exit(1)

    print(f"Successfully read {len(x_coords)} data points.")

    # Convert lists to NumPy arrays for plotting efficiency
    x = np.array(x_coords)
    y = np.array(y_coords)
    z = np.array(values) # Use 'z' for the height axis

    # --- Create the 3D Plot ---
    fig = plt.figure(figsize=(10, 7)) # Adjust figure size if needed
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    # 'c=z' uses the value (z) to determine the color
    # 'cmap' specifies the color map (viridis, plasma, inferno, magma, jet, etc.)
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Value (Height)')
    ax.set_title('3D Scatter Plot of Data')

    # Add a color bar to show the mapping between color and value
    fig.colorbar(scatter, label='Value')

    # Improve layout
    plt.tight_layout()

    # Display the plot
    print("Displaying plot...")
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    file_to_plot = "VecteurU_b"#input("Enter the path to the data file: ")
    plot_data_from_file(file_to_plot)