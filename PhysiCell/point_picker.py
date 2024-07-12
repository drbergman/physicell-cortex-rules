import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
kw_data = sc.read_h5ad("/Users/davidzhou/Downloads/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad")
isocortex = kw_data.obs[kw_data.obs["cell_label"].isin(kw_data.obsm["atlas"][kw_data.obsm["atlas"]["Isocortex"] == True].index)]

def onclick(event):
    ind = event.ind[0]
    print(ind, isocortex['x'][ind], isocortex['y'][ind])
'''    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind

    print("Coords?", ind, x, y)
    print('Artist picked:', event.artist)
    print('{} vertices picked'.format(len(ind)))
    print('Pick between vertices {} and {}'.format(min(ind), max(ind)+1))
    print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    print('Data point:', x[ind[0]], y[ind[0]])'''

fig, ax = plt.subplots()

tolerance = 10 # points
ax.scatter(isocortex['x'], -isocortex['y'], s = 5, picker=True)

fig.canvas.mpl_connect('pick_event', onclick)

plt.show()