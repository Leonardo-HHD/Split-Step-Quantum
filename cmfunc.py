import numpy as np
from matplotlib.colors import hsv_to_rgb

def complex_to_rgb(Z):
    """Convert complex values to their rgb equivalent.
    Parameters
    Z : array_like
        The complex values.
    """
    a = np.abs(Z)
    arg = np.angle(Z)

    h = (arg + np.pi)  / (2 * np.pi) # hue
    s = np.ones(h.shape) # saturation
    v = a  / np.amax(a)  # alpha


    c = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1)  ) # --> tuple
    return c

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

def hue_plate():

    theta = np.linspace(0,2*np.pi,360)
    rho = np.linspace(0,1,20)
    theta_mesh,rho_mesh=np.meshgrid(theta,rho)
    hue_data = np.exp(1j*theta_mesh)

    fig, ax = plt.subplots(figsize=(2/2.54, 3/2.54), dpi=300, subplot_kw={'projection':'polar'}, facecolor='None',
                           gridspec_kw=dict(left=0.3, right=0.7, top=0.88, bottom=0.0))
    ax.set_xticks([0, 0.5*np.pi, 1*np.pi, 1.5*np.pi])
    ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$-\pi/2$'])
    ax.set_yticks([])
    ax.set_title(r'$\arg[\Psi]$')
    ax.grid(False)
    # ax.axis(False)
    ax.pcolormesh(theta_mesh, rho_mesh,
        complex_to_rgb(hue_data), shading='auto', rasterized=True
    )
    return fig, ax

if __name__ == "__main__":
    hue_plate()
    plt.show()