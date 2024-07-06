import torch
from mpl_toolkits.mplot3d import Axes3D

def normalize(vector):
    return vector / torch.norm(vector)

def rotation_matrix(axis, theta):
    """
    Rodrigues' rotation formula to compute the rotation matrix.
    axis : tensor, shape (3,)
        The axis around which to rotate.
    theta : float
        The rotation angle in radians.
    """
    axis = normalize(axis)
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def find_perpendicular_vector(v):
    """
    Find a vector that is perpendicular to the given vector v.
    """
    if torch.allclose(v, torch.tensor([0, 0, 1], dtype=torch.float32)):
        return torch.tensor([0, 1, 0], dtype=torch.float32)
    return torch.cross(v, torch.tensor([0, 0, 1], dtype=torch.float32))

def define_new_axis(p1, p2, theta):
    # Calculate midpoint
    M = (p1 + p2) / 2

    # Direction vector of the original axis a1
    d = p2 - p1
    d_normalized = normalize(d)

    # Find a vector perpendicular to d
    perp_vector = find_perpendicular_vector(d_normalized)
    perp_vector_normalized = normalize(perp_vector)

    # Calculate rotation matrix around d by angle theta
    R = rotation_matrix(d_normalized, theta)

    # Rotate the perpendicular vector to get the new direction vector
    new_direction = torch.matmul(R, perp_vector_normalized)
    new_direction = normalize(new_direction)

    return M, new_direction


if __name__ == "__main__":
    # Example usage:
    p1 = torch.tensor([0, 0, 0], dtype=torch.float32)
    p2 = torch.tensor([1, 0, 0], dtype=torch.float32)
    theta =  torch.tensor(0, dtype=torch.float32)   # 45 degrees

    midpoint, new_axis_direction = define_new_axis(p1, p2, theta)
    #plot p1, p2, midpoint, new_axis_direction with matplotlib
    import matplotlib.pyplot as plt

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(p1[0], p1[1], p1[2], color='red', label='p1')
    ax.scatter(p2[0], p2[1], p2[2], color='blue', label='p2')
    ax.scatter(midpoint[0], midpoint[1], midpoint[2], color='green', label='midpoint')

    # Plot the new axis direction as a line
    # ax.plot([midpoint[0], midpoint[0] + new_axis_direction[0]],
    #     [midpoint[1], midpoint[1] + new_axis_direction[1]],
    #     [midpoint[2], midpoint[2] + new_axis_direction[2]], color='orange', label='new_axis_direction')

    thetas = torch.linspace(0, 2*torch.pi, 36)
    print(thetas)
    for theta in thetas:
        midpoint, new_axis_direction = define_new_axis(p1, p2, theta)
        ax.plot([midpoint[0], midpoint[0] + new_axis_direction[0]],
            [midpoint[1], midpoint[1] + new_axis_direction[1]],
            [midpoint[2], midpoint[2] + new_axis_direction[2]], color='orange')
    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()

    print("Midpoint:", midpoint)
    print("New Axis Direction:", new_axis_direction)

    # N = 4
    # p1 = np.zeros((N, 3))
    # p2 = np.zeros((N, 3))
    # p2[:, 0] = np.linspace(0, 1, N)
    # thetas = np.linspace(0, np.pi, N)
    # midpoints, new_axisess = define_new_axis(p1, p2, thetas)
