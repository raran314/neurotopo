import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

def swc_to_graph(swc):
    """
    Convert SWC data to a graph
    """
    G = nx.Graph()
    for i, row in swc.iterrows():
        G.add_node(row['n'], type=row['type'], x=row['x'], y=row['y'], z=row['z'], r=row['r'])
    for i, row in swc.iterrows():
        if row['parent'] != -1:
            G.add_edge(row['n'], row['parent'])
    return G

def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

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

def visualize_curve(ax, curve):
    curve = sort_curve(curve)
    ax.plot(curve[:,0], curve[:,1], curve[:,2], '-', linewidth=0.7, markersize=2)

def visualize_in_window(curves, L):
    linking_numbers = []
    # Find the range of data
    min_x = min([np.min(curve[:, 0]) for curve in curves])
    max_x = max([np.max(curve[:, 0]) for curve in curves])
    min_y = min([np.min(curve[:, 1]) for curve in curves])
    max_y = max([np.max(curve[:, 1]) for curve in curves])
    min_z = min([np.min(curve[:, 2]) for curve in curves])
    max_z = max([np.max(curve[:, 2]) for curve in curves])

    # Loop over windows
    for x in np.arange(min_x, max_x, L):
        print('hi')
        for y in np.arange(min_y, max_y, L):
            for z in np.arange(min_z, max_z, L):
                linking_number_window = 0
                num_pairs = 0
                for i, curve1 in enumerate(curves):
                    for j, curve2 in enumerate(curves):
                        if i != j:
                            curve1_in_window = curve1[(curve1[:, 0] >= x) & (curve1[:, 0] < x + L) &
                                                       (curve1[:, 1] >= y) & (curve1[:, 1] < y + L) &
                                                       (curve1[:, 2] >= z) & (curve1[:, 2] < z + L)]
                            curve2_in_window = curve2[(curve2[:, 0] >= x) & (curve2[:, 0] < x + L) &
                                                       (curve2[:, 1] >= y) & (curve2[:, 1] < y + L) &
                                                       (curve2[:, 2] >= z) & (curve2[:, 2] < z + L)]
                            if len(curve1_in_window) > 0 and len(curve2_in_window) > 0:
                                linking_number_window += gauss_linking_integral(curve1_in_window, curve2_in_window)
                                num_pairs += 1
                if num_pairs > 0:
                    linking_numbers.append(linking_number_window / num_pairs)

    average_linking_number = np.mean(linking_numbers)
    print(f"Average linking number over all connected components within each window: {average_linking_number}")

swc_file_path = '/Users/romyaran/Desktop/Summer 2024/confocal_stack.swc'
dta = pd.read_csv(swc_file_path, delim_whitespace=True, header=None, comment='#', names=['n', 'type', 'x', 'y', 'z', 'r', 'parent'])

G = swc_to_graph(dta)

conncomp = list(nx.connected_components(G))
cc_indices = [list(map(lambda x: x - 1, cc)) for cc in conncomp]
curves = [dta.iloc[cc][['x', 'y', 'z']].values for cc in cc_indices]
visualize_in_window(curves, L=5)  # Adjust the window size L as needed
