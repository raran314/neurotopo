import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_normal_vector(v1, v2):
    return np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))

def dihedral_angle(n1, n2):
    return np.arcsin(np.dot(n1, n2))

def gauss_linking_integral(polygon1, polygon2):
    n = len(polygon1) - 1
    m = len(polygon2) - 1
    L = 0

    for i in range(n):
        for j in range(m):
            p_i = np.array(polygon1[i])
            p_ip1 = np.array(polygon1[i + 1])
            r_j = np.array(polygon2[j])
            r_jp1 = np.array(polygon2[j + 1])

            r_ij = p_i - r_j
            r_ij1 = p_i - r_jp1
            r_i1j = p_ip1 - r_j
            r_i1j1 = p_ip1 - r_jp1

            n1 = compute_normal_vector(r_ij, r_ij1)
            n2 = compute_normal_vector(r_ij1, r_i1j1)
            n3 = compute_normal_vector(r_i1j1, r_i1j)
            n4 = compute_normal_vector(r_i1j, r_ij)

            area_Qij = (
                dihedral_angle(n1, n2) +
                dihedral_angle(n2, n3) +
                dihedral_angle(n3, n4) +
                dihedral_angle(n4, n1)
            )

            L += area_Qij / (4 * np.pi)

    return L

def plot_chains(polygon1, polygon2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    polygon1 = np.array(polygon1)
    polygon2 = np.array(polygon2)

    ax.plot(polygon1[:, 0], polygon1[:, 1], polygon1[:, 2], label='Polygonal Chain 1', color='b')
    ax.plot(polygon2[:, 0], polygon2[:, 1], polygon2[:, 2], label='Polygonal Chain 2', color='r')

    ax.scatter(polygon1[:, 0], polygon1[:, 1], polygon1[:, 2], color='b')
    ax.scatter(polygon2[:, 0], polygon2[:, 1], polygon2[:, 2], color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

# Example of more complex 3D polygonal chains
polygon1 = [
    [0, 0, 0],
    [1, 2, 1],
    [2, 3, 0],
    [3, 2, -1],
    [4, 0, 0]
]

polygon2 = [
    [0, 0, 1],
    [1, 2, 2],
    [2, 3, 1],
    [3, 2, 0],
    [4, 0, 1]
]

linking_number = gauss_linking_integral(polygon1, polygon2)
print(f"The finite form of the Gauss linking integral is: {linking_number}")

# Plot the polygonal chains
plot_chains(polygon1, polygon2)
