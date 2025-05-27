# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: particles.py
PROJECT: Split-Step Quantum
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.0
CREATED: 2025-05-21
LAST MODIFIED: 2025-05-24

DESCRIPTION:
This script implements the particles.
"""
import numpy as np
from numpy import array, asarray, stack, dot, cross, einsum
from numpy import eye, transpose, ones, full
from numpy import real, imag, conj, angle, iscomplexobj, isrealobj
from numpy import abs, sign, inf, pi, exp, log, sin, asin, cos, acos, tan, atan2, sinh, asinh, cosh, acosh, tanh, atanh
from numpy.linalg import norm

from scipy.constants import c, epsilon_0 as esp_0, mu_0, h, hbar, e, alpha, N_A, k as kB
from scipy.constants import m_e, m_p, m_n, m_u # [kg]
from scipy.constants import eV, erg # [J]
from scipy.constants import angstrom, fermi # [m]
from scipy.constants import milli, micro, nano, pico, femto, atto, zepto, yocto

if __name__ == "__main__":
    from coords import cartesian_to_cylindrical
    from wavefunc import gaussian_packet, Landau_eigenket, Landau_eigenket_bispinor
else:
    from .coords import cartesian_to_cylindrical
    from .wavefunc import gaussian_packet, Landau_eigenket, Landau_eigenket_bispinor

from abc import ABC, abstractmethod
class particle(ABC):
    m = None
    q = None
    s = None
    g = None
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def __init__(self, type_name=None, **kwargs):
        if type_name == None:
            print("Reminder: quantum state undefined.")
        else:
            self.set_state(type_name,**kwargs)
    @abstractmethod
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        pass
class electron(particle):
    m = m_e
    q = -e
    s = 1/2
    g = -2.002319304360
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        if type_name == 'Gaussian':
            self.wx = kwargs['wx']
            self.wy = kwargs['wy']
            self.wz = kwargs['wz']
            # self.px = kwargs['px']
            # self.py = kwargs['py']
            # self.pz = kwargs['pz']
            cartesian_wavefunc = gaussian_packet(
                x0=self.x0, y0=self.y0, z0=self.z0,
                wx=self.wx, wy=self.wy, wz=self.wz,
                px=self.px, py=self.py, pz=self.pz,
            )
            self.wavefunc = cartesian_wavefunc
        elif type_name == 'Landau':
            self.n = kwargs['n']
            self.ell = kwargs['ell']
            self.Bz = kwargs['Bz']
            # self.pz = kwargs['pz']
            self.wz = kwargs['wz']
            r0 = cartesian_to_cylindrical(x0,y0,z0)
            wavefunc_cylinderic = Landau_eigenket(
                n=self.n,ell=self.ell,Bz=self.Bz,
                wz=self.wz,pz=self.pz,r0=r0
                )
            wavefunc_cartesian = lambda x,y,z: wavefunc_cylinderic(*(cartesian_to_cylindrical(x-x0,y-y0,z-z0)))
            self.wavefunc = wavefunc_cartesian
        else:
            raise NotImplementedError(f"State type '{type_name}' is not implemented.")
class positron(particle):
    m = m_e
    q = +e
    s = 1/2
    g = +2.002319304360
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        if type_name == 'Gaussian':
            self.wx = kwargs['wx']
            self.wy = kwargs['wy']
            self.wz = kwargs['wz']
            # self.px = kwargs['px']
            # self.py = kwargs['py']
            # self.pz = kwargs['pz']
            cartesian_wavefunc = gaussian_packet(
                x0=self.x0, y0=self.y0, z0=self.z0,
                wx=self.wx, wy=self.wy, wz=self.wz,
                px=self.px, py=self.py, pz=self.pz,
            )
            self.wavefunc = cartesian_wavefunc
        elif type_name == 'Landau':
            self.n = kwargs['n']
            self.ell = kwargs['ell']
            self.Bz = kwargs['Bz']
            # self.pz = kwargs['pz']
            self.wz = kwargs['wz']
            r0 = cartesian_to_cylindrical(x0,y0,z0)
            wavefunc_cylinderic = Landau_eigenket(
                n=self.n,ell=self.ell,Bz=self.Bz,
                wz=self.wz,pz=self.pz,r0=r0
                )
            wavefunc_cartesian = lambda x,y,z: wavefunc_cylinderic(*(cartesian_to_cylindrical(x+x0,y+y0,z+z0)))
            self.wavefunc = wavefunc_cartesian
        else:
            raise NotImplementedError(f"State type '{type_name}' is not implemented.")
class proton(particle):
    m = m_p
    q = +e
    s = 1/2
    g = +5.58569468
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        if type_name == 'Gaussian':
            self.wx = kwargs['wx']
            self.wy = kwargs['wy']
            self.wz = kwargs['wz']
            # self.px = kwargs['px']
            # self.py = kwargs['py']
            # self.pz = kwargs['pz']
            cartesian_wavefunc = gaussian_packet(
                x0=self.x0, y0=self.y0, z0=self.z0,
                wx=self.wx, wy=self.wy, wz=self.wz,
                px=self.px, py=self.py, pz=self.pz,
            )
            self.wavefunc = cartesian_wavefunc
        elif type_name == 'Landau':
            self.n = kwargs['n']
            self.ell = kwargs['ell']
            self.Bz = kwargs['Bz']
            # self.pz = kwargs['pz']
            self.wz = kwargs['wz']
            r0 = cartesian_to_cylindrical(x0,y0,z0)
            wavefunc_cylinderic = Landau_eigenket(
                n=self.n,ell=self.ell,Bz=self.Bz,
                wz=self.wz,pz=self.pz,r0=r0
                )
            wavefunc_cartesian = lambda x,y,z: wavefunc_cylinderic(*(cartesian_to_cylindrical(x+x0,y+y0,z+z0)))
            self.wavefunc = wavefunc_cartesian
        else:
            raise NotImplementedError(f"State type '{type_name}' is not implemented.")
class antiproton(particle):
    m = m_p
    q = -e
    s = 1/2
    g = -5.58569468
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        if type_name == 'Gaussian':
            self.wx = kwargs['wx']
            self.wy = kwargs['wy']
            self.wz = kwargs['wz']
            # self.px = kwargs['px']
            # self.py = kwargs['py']
            # self.pz = kwargs['pz']
            cartesian_wavefunc = gaussian_packet(
                x0=self.x0, y0=self.y0, z0=self.z0,
                wx=self.wx, wy=self.wy, wz=self.wz,
                px=self.px, py=self.py, pz=self.pz,
            )
            self.wavefunc = cartesian_wavefunc
        elif type_name == 'Landau':
            self.n = kwargs['n']
            self.ell = kwargs['ell']
            self.Bz = kwargs['Bz']
            # self.pz = kwargs['pz']
            self.wz = kwargs['wz']
            r0 = cartesian_to_cylindrical(x0,y0,z0)
            wavefunc_cylinderic = Landau_eigenket(
                n=self.n,ell=self.ell,Bz=self.Bz,
                wz=self.wz,pz=self.pz,r0=r0
                )
            wavefunc_cartesian = lambda x,y,z: wavefunc_cylinderic(*(cartesian_to_cylindrical(x+x0,y+y0,z+z0)))
            self.wavefunc = wavefunc_cartesian
        else:
            raise NotImplementedError(f"State type '{type_name}' is not implemented.")
class alpha(particle):
    m = 6.6446573450e-27 # kg
    q = +2*e
    s = 0
    g = None
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        if type_name == 'Gaussian':
            self.wx = kwargs['wx']
            self.wy = kwargs['wy']
            self.wz = kwargs['wz']
            # self.px = kwargs['px']
            # self.py = kwargs['py']
            # self.pz = kwargs['pz']
            cartesian_wavefunc = gaussian_packet(
                x0=self.x0, y0=self.y0, z0=self.z0,
                wx=self.wx, wy=self.wy, wz=self.wz,
                px=self.px, py=self.py, pz=self.pz,
            )
            self.wavefunc = cartesian_wavefunc
        else:
            raise NotImplementedError(f"State type '{type_name}' is not implemented.")
class neutron(particle):
    m = m_n
    q = 0
    s = 1/2
    g = -3.826085
    x0, y0, z0 = None, None, None
    px, py, pz = None, None, None
    state_type = None
    wavefunc = None
    def set_state(self, type_name:str,
                  x0=0, y0=0, z0=0,
                  px=0, py=0, pz=0,
                  **kwargs):
        self.x0 = x0 #kwargs['x0']
        self.y0 = y0 #kwargs['y0']
        self.z0 = z0 #kwargs['z0']
        self.px = px #kwargs['px']
        self.py = py #kwargs['py']
        self.pz = pz #kwargs['pz']
        self.state_type = type_name
        if type_name == 'Gaussian':
            self.wx = kwargs['wx']
            self.wy = kwargs['wy']
            self.wz = kwargs['wz']
            # self.px = kwargs['px']
            # self.py = kwargs['py']
            # self.pz = kwargs['pz']
            cartesian_wavefunc = gaussian_packet(
                x0=self.x0, y0=self.y0, z0=self.z0,
                wx=self.wx, wy=self.wy, wz=self.wz,
                px=self.px, py=self.py, pz=self.pz,
            )
            self.wavefunc = cartesian_wavefunc
        else:
            raise NotImplementedError(f"State type '{type_name}' is not implemented.")

particle_dict = {
    'electron': electron,
    'positron': positron,
    'proton': proton,
    'antiproton': antiproton,
    'alpha': alpha,
    'neutron': neutron,
}
def generate_N_body_system(particle_list:list,**kwargs):
    N = len(particle_list)
    particles = []
    for i in range(N):
        particle_type = particle_list[i].lower()
        if particle_type not in particle_dict:
            raise ValueError(f"Particle '{particle_list[i]}' is not defined.\n Please check if in the name-list of supporting particles:\n {list(particle_dict.keys())}.")
        else:
            particles += [particle_dict[particle_type]()]
    # Np = len(particles)
    return particles#, Np
#%% test
def test1():
    part1, part2, part3, part4, part5, part6 = electron(), electron(), positron(), proton(), antiproton(), alpha()
    part1.set_state(
        type_name='Gaussian',
        x0=0,
        y0=0,
        z0=0,
        wx=25*nano,
        wy=25*nano,
        wz=25*nano,
        px=0,
        py=0,
        pz=0,
    )
    part2.set_state(
        type_name='Landau',
        x0=0,
        y0=0,
        z0=0,
        n=0,
        ell=-1,
        Bz=1.9,
        pz=0,
        wz=25*nano,
    )

    x = np.linspace(-200*nano, 200*nano, 100)
    y = np.linspace(-200*nano, 200*nano, 100)
    z = 0
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Psi_0 = part1.wavefunc(X,Y,Z)
    Psi_1 = part2.wavefunc(X,Y,Z)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    znode = 0
    rho_0 = abs(Psi_0)**2
    plt.subplot(121)
    plt.title('Psi_0')
    plt.gca().set_box_aspect(1)
    plt.pcolormesh(X[:,:,znode],Y[:,:,znode],rho_0[:,:,znode],
                cmap='bone', shading='gouraud')
    plt.colorbar(orientation='horizontal')
    rho_1 = abs(Psi_1)**2
    plt.subplot(122)
    plt.title('Psi_1')
    plt.gca().set_box_aspect(1)
    plt.pcolormesh(X[:,:,znode],Y[:,:,znode],rho_1[:,:,znode],
                cmap='bone', shading='gouraud')
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()
    plt.close()
def test2():
    Np = 4
    # particles = np.full(Np, electron())
    particles = []
    for _ in range(Np):
        particles += [electron()]
    for i in np.arange(Np):
        print(f"particle {i}")
        particles[i].set_state(type_name="Landau",
                n=i,ell=(-1)**i,Bz=1.9,pz=0,wz=25*nano)
        print(f"    n={i},ell={(-1)**i},Bz=1.9T,pz=0,wz=25nm", particles[i].wavefunc)
    for p in particles:
        print(p.wavefunc)

    x = np.linspace(-200*nano, 200*nano, 100)
    y = np.linspace(-200*nano, 200*nano, 100)
    z = 0
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    znode =0

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6*Np,8))
    for i in np.arange(Np):
        print(f"particle {i}: n={particles[i].n}, ell={particles[i].ell}.", particles[i].wavefunc)
        Psi = particles[i].wavefunc(X,Y,Z)
        rho = abs(Psi)**2
        plt.subplot(1,4,i+1)
        plt.title(f'Psi_{i}')
        plt.gca().set_box_aspect(1)
        plt.pcolormesh(X[:,:,znode],Y[:,:,znode],rho[:,:,znode],
                    cmap='bone', shading='gouraud')
        plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    test1()
    test2()