import scipy.io as sio

# Load the .mat file
mat_contents = sio.loadmat('E:\\gazenet\\MPIIFaceGaze\\MPIIFaceGaze\\p00\\Calibration\\Camera.mat')

# Print the keys (variable names) in the .mat file
print("Variables in the .mat file:")
print(mat_contents.keys())

# Print the contents of each variable
for key in mat_contents:
    if key.startswith('__'):  # Skip metadata
        continue
    print(f"\nVariable: {key}")
    print(mat_contents[key])