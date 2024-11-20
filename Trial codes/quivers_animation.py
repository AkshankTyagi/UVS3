import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
from IPython.display import HTML
from matplotlib.animation import PillowWriter

random.seed(20210110)

repo = (pd.DataFrame({'x':np.random.randint(0,5, size=50),
                     'y':np.random.randint(0,5, size=50),
                     'z':np.random.randint(0,5, size=50)}))
data = np.array((repo['x'].values, repo['y'].values, repo['z'].values))

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

quiver = ax.quiver([],[],[],[],[],[])
print (type(quiver))

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

def update(i):
    global quiver
    quiver.remove()
    quiver = ax.quiver(0,0,0,data[0][i],data[1][i],data[2][i])

ani = FuncAnimation(fig, update, frames=50, interval=50)
plt.show()
# ani.save('./quiver_test_ani.gif', writer='pillow')
plt.close()
# jupyter lab
# HTML(ani.to_html5_video())