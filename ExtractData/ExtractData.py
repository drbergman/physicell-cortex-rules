import anndata
import os
import matplotlib.pyplot as plt
import numpy as np

os.getcwd()

# path_to_file = "./ExtractData/data/Zhuang-ABCA-1-raw_wMeta_wAnnotations_wAtlas_sub5_KW.h5ad"
# path_to_file = "./ExtractData/data/Zhuang-ABCA-1-1.064_raw_wClusterAnnots.h5ad"
path_to_file = "./ExtractData/data/Zhuang-ABCA-1-raw_1.086_wMeta_wAnnotations_KW.h5ad"

region_name = 'AUD'


adata = anndata.read_h5ad(path_to_file)
adata
adata.obs
# get names of the adata.obs dataframe
adata.obs.columns
adata.obs.cluster_alias # no? just a bunch of ints
adata.obs.CCF_cell_label # int labels
adata.obs.cell_label # more labels
adata.obs.brain_section_label_x # files the cells come from? also with some spatial info in the name
adata.obs.brain_section_label_x.unique()

adata.obsm
adata.obsm['atlas'] # big ol' matrix/dataframe
adata.obsm['atlas'].columns

count_by_column = {}
for col_name in adata.obsm['atlas'].columns:
    count_by_column[col_name] = sum(adata.obsm['atlas'][col_name])
count_by_column
# let's go with AUD (for auditory cortex?)

if region_name not in adata.obsm['atlas'].columns:
    raise ValueError(f"Region {region_name} not found in atlas")
if count_by_column[region_name] == 0:
    raise ValueError(f"Region {region_name} has 0 cells in this dataset")

adata.obsm['atlas'][region_name]
cell_in_region = adata.obsm['atlas'][region_name]
sum(cell_in_region)

# get indices for each layer
cell_in_layer = {}
counts_in_layer = {}
layer_cols = [col_name for col_name in adata.obsm['atlas'].columns if col_name.startswith('layer')]
for layer_name in layer_cols:
    cell_in_layer[layer_name] = cell_in_region & adata.obsm['atlas'][layer_name]
    counts_in_layer[layer_name] = sum(cell_in_layer[layer_name])

print(counts_in_layer)

# make a plot of the cells in this region colored by layer
x = adata.obs.x
y = adata.obs.y
z = adata.obs.z

fig, ax = plt.subplots()
for layer_name in layer_cols:
    ax.scatter(x[cell_in_layer[layer_name]], y[cell_in_layer[layer_name]], label=layer_name)

ax.legend()
plt.show(block=False)

# let's try to plot the cells in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for layer_name in layer_cols:
    ax.scatter(x[cell_in_layer[layer_name]], y[cell_in_layer[layer_name]], z[cell_in_layer[layer_name]], label=layer_name)

ax.legend()
plt.show(block=False) # see that there's three z slices here

# pick the top z slice
z[cell_in_region].unique() # 6.88164308, 7.00373086, 7.12315029 (either AUD or MO)

cell_in_top_slice_region = cell_in_region & (z == z[cell_in_region].max())

# check with plot
fig, ax = plt.subplots()
for layer_name in layer_cols:
    ax.scatter(x[cell_in_top_slice_region & cell_in_layer[layer_name]], y[cell_in_top_slice_region & cell_in_layer[layer_name]], label=layer_name)

ax.legend()
plt.show(block=False)

# get the x,y coords for all these cells
xx = x[cell_in_top_slice_region]
yy = y[cell_in_top_slice_region]

# get convex hull of these
from scipy.spatial import ConvexHull
points = np.array([xx, yy]).T
ch = ConvexHull(points)

# add convex hull to plot
fig, ax = plt.subplots()
for layer_name in layer_cols:
    ax.scatter(x[cell_in_top_slice_region & cell_in_layer[layer_name]], y[cell_in_top_slice_region & cell_in_layer[layer_name]], label=layer_name)

for simplex in ch.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.plot(points[ch.vertices, 0], points[ch.vertices, 1], 'ro')

ax.legend()
plt.show(block=False)

if region_name == 'AUD':
    if path_to_file == "./ExtractData/data/Zhuang-ABCA-1-raw_wMeta_wAnnotations_wAtlas_sub5_KW.h5ad":
        # select point at top-right of convex hull
        p1_ind = (points[:,0] > 1.726) & (points[:,1] > 3.620)
        p1_x = points[p1_ind,0][0]
        p1_y = points[p1_ind,1][0]
        # select point at bottom-right of convex hull
        p2_ind = (points[:,0] > 1.966)
        p2_x = points[p2_ind,0][0]
        p2_y = points[p2_ind,1][0]
    elif path_to_file == "./ExtractData/data/Zhuang-ABCA-1-raw_1.086_wMeta_wAnnotations_KW.h5ad":
        p1_ind = (points[:,0] > 1.78) & (points[:,1] > 3.675)
        p1_x = points[p1_ind,0][0]
        p1_y = points[p1_ind,1][0]
        # select point at bottom-right of convex hull
        p2_ind = (points[:,0] > 2.290) & (points[:,1] > 2.89)
        p2_x = points[p2_ind,0][0]
        p2_y = points[p2_ind,1][0]
        dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
        ideal_dist = 0.9
        scale = ideal_dist / dist
        c = (scale - 1) * (-0.5)
        p1_x = p1_x + c * (p2_x - p1_x)
        p1_y = p1_y + c * (p2_y - p1_y)
        # recall that p1_x and p2_x have already been moved to their final positions...
        temp_dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
        p2_x = p1_x + (p2_x - p1_x) * ideal_dist / temp_dist
        p2_y = p1_y + (p2_y - p1_y) * ideal_dist / temp_dist
elif region_name == 'MO':
    # at top-left but call it tr to match AUD
    p1_ind = (points[:,0] > 4.592) & (points[:,0] < 4.594) & (points[:,1] > 2.368)
    p1_x = points[p1_ind,0][0]
    p1_y = points[p1_ind,1][0]-0.08
    # select point at top-right of convex hull but call it br to match AUD
    p2_ind = (points[:,0] > 4.8475) & (points[:,1] > 2.30)
    p2_x = points[p2_ind,0][0]
    p2_y = points[p2_ind,1][0]
    dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
    scale = 0.6 / dist
    p1_x = p2_x + (p1_x - p2_x) * scale
    p1_y = p2_y + (p1_y - p2_y) * scale

# get slope between them
dx = p2_x - p1_x
dy = p2_y - p1_y

def between_perp_lines(points, p1, p2):
    s1 = np.sign((points - p1) @ (p2 - p1))
    s2 = np.sign((points - p2) @ (p2 - p1))
    return s1 * s2 <= 0 # if they have different signs or one is 0, then they are between/on the lines

# plot these
fig, ax = plt.subplots()
x_slice = x[cell_in_top_slice_region]
y_slice = y[cell_in_top_slice_region]
for layer_name in layer_cols:
    x_layer = x[cell_in_top_slice_region & cell_in_layer[layer_name]]
    y_layer = y[cell_in_top_slice_region & cell_in_layer[layer_name]]
    between_lines = between_perp_lines(np.array([x_layer, y_layer]).T, np.array([p1_x, p1_y]), np.array([p2_x, p2_y]))
    ax.scatter(x_layer[between_lines], y_layer[between_lines], label=layer_name)
    ax.scatter(x_layer[~between_lines], y_layer[~between_lines], label=None, color='black')

# plot the convex hull
for simplex in ch.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.plot(points[ch.vertices, 0], points[ch.vertices, 1], 'ro')

# plot the line between tr and br
plt.plot([p1_x, p2_x], [p1_y, p2_y], 'r--')
xL = plt.xlim()
yL = plt.ylim()
# plot the perpeindicular lines
for p in [[p1_x, p1_y], [p2_x, p2_y]]:
    c_x = - (xL - p[0]) / dy
    c_x.sort() # negative first, then positive (each direction along perp line)
    c_y = (yL - p[1]) / dx
    c_y.sort()
    c_1 = max(c_x[0], c_y[0])
    c_2 = min(c_x[1], c_y[1])
    plt.plot([p[0] - c_1*dy, p[0] - c_2*dy], [p[1] + c_1*dx, p[1] + c_2*dx], 'r--')
    
ax.legend()
plt.show(block=False)

# count the cells in each layer between the lines
layer_count_between_lines = {}
for layer_name in layer_cols:
    x_layer = x[cell_in_top_slice_region & cell_in_layer[layer_name]]
    y_layer = y[cell_in_top_slice_region & cell_in_layer[layer_name]]
    between_lines = between_perp_lines(np.array([x_layer, y_layer]).T, np.array([p1_x, p1_y]), np.array([p2_x, p2_y]))
    layer_count_between_lines[layer_name] = sum(between_lines)

layer_count_between_lines # {1: 61, 2/3: 196, 4: 53, 5: 214, 6: 197} for AUD in 1.083
layer_count_between_lines # {1: 118, 2/3: 338, 4: 121, 5: 362, 6: 220} for AUD in 1.086

sum(layer_count_between_lines.values())

# add these counts to the plot
fig, ax = plt.subplots()
for layer_name in layer_cols:
    x_layer = x[cell_in_top_slice_region & cell_in_layer[layer_name]]
    y_layer = y[cell_in_top_slice_region & cell_in_layer[layer_name]]
    between_lines = between_perp_lines(np.array([x_layer, y_layer]).T, np.array([p1_x, p1_y]), np.array([p2_x, p2_y]))
    ax.scatter(x_layer[between_lines], y_layer[between_lines], label=layer_name)
    ax.scatter(x_layer[~between_lines], y_layer[~between_lines], label=None, color='black')

# plot the convex hull
for simplex in ch.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.plot(points[ch.vertices, 0], points[ch.vertices, 1], 'ro')

# plot the line between tr and br
plt.plot([p1_x, p2_x], [p1_y, p2_y], 'r--')
xL = plt.xlim()
yL = plt.ylim()
# plot the perpeindicular lines
for p in [[p1_x, p1_y], [p2_x, p2_y]]:
    c_x = - (xL - p[0]) / dy
    c_x.sort() # negative first, then positive (each direction along perp line)
    c_y = (yL - p[1]) / dx
    c_y.sort()
    c_1 = max(c_x[0], c_y[0])
    c_2 = min(c_x[1], c_y[1])
    plt.plot([p[0] - c_1*dy, p[0] - c_2*dy], [p[1] + c_1*dx, p[1] + c_2*dx], 'r--')

# add text with counts
for layer_name in layer_cols:
    x_layer = x[cell_in_top_slice_region & cell_in_layer[layer_name]]
    y_layer = y[cell_in_top_slice_region & cell_in_layer[layer_name]]
    between_lines = between_perp_lines(np.array([x_layer, y_layer]).T, np.array([p1_x, p1_y]), np.array([p2_x, p2_y]))
    ax.text(x_layer[between_lines].mean(), y_layer[between_lines].mean(), str(sum(between_lines)), fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

ax.legend()
plt.show(block=False)

# get distance between the two points
dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
dist * 1000 # assume units are in mm ~600um

# check meta data of adata for units of x,y,z
adata.uns.keys()
adata

# plot all cells here, highlghting those in the AUD region
fig, ax = plt.subplots()
for layer_name in layer_cols:
    ax.scatter(x[cell_in_region & cell_in_layer[layer_name]], y[cell_in_region & cell_in_layer[layer_name]], 30, label=f"{region_name}: {layer_name}")

for layer_name in layer_cols:
    ax.scatter(x[( ~cell_in_region) & adata.obsm['atlas'][layer_name]], y[(~cell_in_region) & adata.obsm['atlas'][layer_name]], 1, label=layer_name, alpha=0.2)

ax.legend()
plt.show(block=False)

# plot them all again but in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for layer_name in layer_cols:
    ax.scatter(x[cell_in_region & cell_in_layer[layer_name]], y[cell_in_region & cell_in_layer[layer_name]], 0.02 + z[cell_in_region & cell_in_layer[layer_name]], label=f"{region_name}: {layer_name}")

for layer_name in layer_cols:
    ax.scatter(x[( ~cell_in_region) & adata.obsm['atlas'][layer_name]], y[(~cell_in_region) & adata.obsm['atlas'][layer_name]], z[(~cell_in_region) & adata.obsm['atlas'][layer_name]], label=layer_name, alpha=0.2)

ax.legend()
plt.show(block=False)