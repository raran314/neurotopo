# %%
import numpy as np
import matplotlib.pyplot as plt
import importlib
import filamentFields
importlib.reload(filamentFields)

filaments = []
for i in range(10):
    x = np.cumsum(np.random.randn(100))
    y = np.cumsum(np.random.randn(100))
    z = np.cumsum(np.random.randn(100))
    x = np.convolve(x, np.ones(5)/5, mode='valid')
    y = np.convolve(y, np.ones(5)/5, mode='valid')
    z = np.convolve(z, np.ones(5)/5, mode='valid')        
    filaments.append(np.vstack([x,y,z]).T)
    

    
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for r in filaments:
    ax.plot(r[:,0],r[:,1],r[:,2])

# %%
fF = filamentFields.filamentFields(filaments)
all_nodes = fF.return_all_nodes()

# %%
query_points = np.array([0,0,0])
R_omega = 2
local_edges = fF.analyzeLocalVolume(query_points, R_omega, 0.01)
# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in range(len(local_edges)):
    edge = local_edges[i]
    p1 = edge[:3]
    p2 = edge[3:]
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])
# %%
# my sampling
filament_edges_list = fF.return_filament_nodes_list()
all_edges = fF.return_all_edges()
# %%
len(filament_edges_list)
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for edge in all_edges:
    p1 = edge[:3]
    p2 = edge[3:]
    
    if (np.linalg.norm(p1 - query_points) < R_omega ) & (np.linalg.norm(p2 - query_points) < R_omega):        
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])
    

# %%

# %%
num_rods = 100
def linspace3(start, stop, num):
    return np.vstack([np.linspace(start[i], stop[i], num) for i in range(3)]).T 

single_edge = linspace3([0,0,0],[0,0,1],10)

mg = np.meshgrid(np.linspace(0,1,num_rods),np.linspace(0,1,num_rods))
mg = [mg[0].flatten(),mg[1].flatten()]
aligned_filaments = []
for i in range(num_rods**2):
    translator = np.array([mg[0][i],mg[1][i],0]).T
    aligned_filaments.append(single_edge + translator)
    
# fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
# for i in range(100):
#     ax.plot(aligned_filaments[i][:,0],aligned_filaments[i][:,1],aligned_filaments[i][:,2])
    
# %%
fF = filamentFields.filamentFields(aligned_filaments)
# %%
fF.analyzeLocalVolume(np.array([0,0,0]), 1,0.01)
