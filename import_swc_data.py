import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is imported

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

swc_file_path = '/Users/romyaran/Desktop/Summer_2024/confocal_stack.swc'
dta = pd.read_csv(swc_file_path, sep='\s+', header=None, comment='#', names=['n', 'type', 'x', 'y', 'z', 'r', 'parent'])

G = swc_to_graph(dta)

conncomp = list(nx.connected_components(G))
cc = np.array(list(conncomp[0]))

curves = []
for cc in conncomp:
    cc = np.array(list(cc))
    curve = dta.iloc[cc-1][['x', 'y', 'z']].values
    curves.append(curve)

def sort_curve(rr):
    centroid = np.mean(rr, axis=0)
    rr_centered = rr - centroid        
    _, _, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0, :]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

fig = plt.figure()
from scipy.ndimage import gaussian_filter1d
sigma = 5
ax = fig.add_subplot(111, projection='3d')
for i in range(len(curves)):
    curve = curves[i]
    curve = sort_curve(curve)
    # smooth out with Gaussian kernel

    curve = gaussian_filter1d(curve, sigma=sigma, axis=0)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], '-', linewidth=0.7, markersize=2)

# Manually set equal aspect ratio
x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()

x_range = x_limits[1] - x_limits[0]
y_range = y_limits[1] - y_limits[0]
z_range = z_limits[1] - z_limits[0]

plot_radius = 0.5 * max([x_range, y_range, z_range])

ax.set_xlim3d([np.mean(x_limits) - plot_radius, np.mean(x_limits) + plot_radius])
ax.set_ylim3d([np.mean(y_limits) - plot_radius, np.mean(y_limits) + plot_radius])
ax.set_zlim3d([np.mean(z_limits) - plot_radius, np.mean(z_limits) + plot_radius])

plt.show()
