import numpy as np

def create_random_rotation_translation_matrix(rotation_range, translation_range):
    # Random rotation angles in radians within the specified range
    theta_x = np.random.uniform(rotation_range[0], rotation_range[1])
    theta_y = np.random.uniform(rotation_range[0], rotation_range[1])
    theta_z = np.random.uniform(rotation_range[0], rotation_range[1])
    
    # Rotation matrices around x, y, and z axes
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Random translation values within the specified range
    tx = np.random.uniform(translation_range[0], translation_range[1])
    ty = np.random.uniform(translation_range[0], translation_range[1])
    tz = np.random.uniform(translation_range[0], translation_range[1])
    
    # Translation matrix
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation matrices
    transformation_matrix = T @ R

    return transformation_matrix


if __name__ == "__main__":
    # Define the range for rotation angles (in radians) and translation distances
    rotation_range = (-np.pi, np.pi)  # full circle range in radians
    translation_range = (-0.5, 0.5)  # translation values range

    # Generate a random 4x4 transformation matrix
    random_matrix = create_random_rotation_translation_matrix(rotation_range, translation_range)
    print("Random 4x4 Rotation and Translation Matrix:\n", random_matrix)
