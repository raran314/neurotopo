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

def visualize_curve(ax, curve):
    curve = sort_curve(curve)
    ax.plot(curve[:,0], curve[:,1], curve[:,2], '-', linewidth=0.7, markersize=2)

def visualize_in_window(curves, L):
    # Find the range of data
    min_x = min([np.min(curve[:, 0]) for curve in curves])
    max_x = max([np.max(curve[:, 0]) for curve in curves])
    min_y = min([np.min(curve[:, 1]) for curve in curves])
    max_y = max([np.max(curve[:, 1]) for curve in curves])
    min_z = min([np.min(curve[:, 2]) for curve in curves])
    max_z = max([np.max(curve[:, 2]) for curve in curves])

    # Loop over windows
    for x in np.arange(min_x, max_x, L):
        for y in np.arange(min_y, max_y, L):
            for z in np.arange(min_z, max_z, L):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlim(x, x + L)
                ax.set_ylim(y, y + L)
                ax.set_zlim(z, z + L)
                for curve in curves:
                    curve_in_window = curve[(curve[:, 0] >= x) & (curve[:, 0] < x + L) &
                                             (curve[:, 1] >= y) & (curve[:, 1] < y + L) &
                                             (curve[:, 2] >= z) & (curve[:, 2] < z + L)]
                    if len(curve_in_window) > 0:
                        visualize_curve(ax, curve_in_window)
                plt.show()

swc_file_path = '/Users/romyaran/Desktop/Summer 2024/confocal_stack.swc'
dta = pd.read_csv(swc_file_path, delim_whitespace=True, header=None, comment='#', names=['n', 'type', 'x', 'y', 'z', 'r', 'parent'])

G = swc_to_graph(dta)

conncomp = list(nx.connected_components(G))
# Convert connected components sets to indices
cc_indices = [np.array(list(cc)) - 1 for cc in conncomp]

# Extract curves based on indices
curves = [dta.iloc[cc][['x', 'y', 'z']].values for cc in cc_indices]

visualize_in_window(curves, L=50)  # Adjust the window size L as needed


