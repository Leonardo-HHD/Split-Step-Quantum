# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: fields.py
PROJECT: Quantum Dynamics in External Fields
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.0
CREATED: 2025-04-13
LAST MODIFIED: 2025-04-29

DESCRIPTION:
This script implements the wave-functions.
"""

#%% Import libraries, functions and constants
import numpy as np
from numpy import array, asarray, stack
from numpy import pi, exp, log, sin, cos, tan, atan, sinh, cosh, tanh, atanh, real, imag, sign, conj
from scipy.constants import c, h, hbar, e, m_e, m_p, epsilon_0 as esp_0, mu_0, k as kB, eV, angstrom, milli, micro, nano, pico, femto, atto
from scipy.special import hermite, factorial, eval_genlaguerre

if __name__ == "__main__":
    from relativisity import gamma
else:
    from .relativisity import gamma

def prob_dens(Psi_arr, type='scalar'):
    '''
    Computes the probability density of the wavefunction.
    '''
    type_list = ['scalar','bispinor']
    if type == 'scalar': # scalar field
        return abs(Psi_arr**2)
    elif type == 'bispinor': # vector field
        return real(np.einsum('...i,...i->...', Psi_arr.conj(), Psi_arr)).astype(np.float32)
    else:
        raise ValueError(f"Invalid type '{type}'. Expected one of {type_list}.")

def Cnl(n, ell):
    '''
    Computes the normalization constant for the Laguerre-Gaussian wave-packet.
    '''
    return (factorial(n) / (factorial(n + abs(ell))))

from abc import ABC,abstractmethod

class cartesian_wavefunc(ABC):

    def __call__(self,x,y,z):
        return self.Psi(x,y,z)

    @abstractmethod
    def Psi(self,x,y,z):
        '''
        Cartesian coords
            x: x coord
            y: y coord
            z: z coord
        '''
        pass

    @staticmethod
    def _test():
        pass

class cylindrical_wavefunc(ABC):

    def __call__(self,rho,theta,z):
        return self.Psi(rho,theta,z)

    @abstractmethod
    def Psi(self,rho,theta,z):
        '''
        Cylindrical coords
            rho: radial coord
            theta: azimuth angle
            z: axial coord
        '''
        pass

    @staticmethod
    def _test():
        pass

class gaussian_packet(cartesian_wavefunc):
    '''
    Gaussian wave-packet: free electron's wavefunction.
    '''
    def __init__(self,wx:float,wy:float,wz:float,px:float,py:float,pz:float,x0=0,y0=0,z0=0):
        '''
        (wx,wy,wz): [m]     packet width (variance)
        (px,py,pz): [N.s]   initial momentum (expectation value)
        (x0,y0,z0): [m]     initial position (expectation value)
        '''
        self.wx = abs(float(wx))
        self.wy = abs(float(wy))
        self.wz = abs(float(wz))
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.z0 = float(z0)

    def Psi(self,x,y,z):
        x,y,z = asarray(x),asarray(y),asarray(z)
        wx,wy,wz=self.wx,self.wy,self.wz
        px,py,pz=self.px,self.py,self.pz
        x0,y0,z0=self.x0,self.y0,self.z0
        x, y, z = x-x0, y-y0, z-z0
        Ampli = ((pi * wx**0.5 * pi * wy**0.5 * pi * wz**0.5)**-0.5)
        Expon = exp(-0.5*(x/wx)**2 - 0.5*(y/wy)**2 - 0.5*(z/wz)**2)
        phase = (px*x + py*y + pz*z)/hbar
        return Ampli * Expon * exp(1j*phase)

    def _test():
        wx, wy, wz = 1e-9, 1e-9, 1e-9
        px, py, pz = 1e-24, 1e-24, 1e-24
        x0, y0, z0 = 1e-9, 1e-9, 0
        Psi = gaussian_packet(wx, wy, wz, px, py, pz, x0, y0, z0)
        x = np.linspace(-5e-9, +5e-9, 100)
        y = np.linspace(-5e-9, +5e-9, 100)
        z = np.linspace(-5e-9, +5e-9, 100)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
        data = Psi.Psi(x_mesh, y_mesh, z_mesh)
        prob_density = prob_dens(data, type='scalar')
        znode = 50
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 8))
        plt.gca().set_box_aspect(1)
        plt.imshow(prob_density[:,:,znode], extent=(-5e-9, +5e-9, -5e-9, +5e-9), origin='lower')
        plt.colorbar(orientation='horizontal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Gaussian Wave-Packet Probability Density')
        plt.show()

if __name__ == "__main__":
    gaussian_packet._test()

class LG_nl_beam(cylindrical_wavefunc):
    '''
    Laguerre-Gaussian beam: vortex electrons' wavefunction propagating along an axial uniform magnetic field.
    References:
    [1] Zou, L.-P. and Zhang, P.-M. and Silenko, A. J. 2021 PRA [10.1103/PhysRevA.103.L010201]
    '''
    def __init__(self,n:int,ell:int,w0:float,vz:float,Bz:float):
        '''
        n: [uint]   radial index
        ell: [int]  topological charge
        w0: [m]     beam waist
        vz: [m/s]   velocity of the beam
        Bz: [T]     magnetic field
        '''

        self.n = abs(int(n))
        self.ell = int(ell)
        self.w0 = float(w0)
        self.vz = float(vz)
        self.Bz = float(Bz)

        kz = m_e*gamma(vz)*vz/hbar #[1/m] wavenumber
        zR = 0.5*kz*w0**2 #[m] Rayleigh distance
        wm = 2*(hbar/abs(e*Bz))**0.5 #[m] magnetic length scale
        zm = 0.5*kz*wm**2 #[m] reduced Larmor distance
        # zL = 2*pi*zm #Larmor distance

        self.kz = kz
        self.zR = zR
        self.wm = wm
        self.zm = zm

    # def __call__(self,rho,theta,z):
    #     return self.Psi(rho,theta,z)

    def w(self,z):
        '''
        z: propagation distance
        kz: wavenumber
        w0: beam waist
        '''
        w0 = self.w0
        zm = self.zm
        zR = self.zR
        wz = w0*(cos(z/zm)**2+(sin(z/zm)*zm/zR)**2)**0.5
        return wz

    @staticmethod
    def C_nl(n,ell):
        Cnl = ((2*factorial(n))/(pi*factorial(n+abs(ell))))**0.5
        return Cnl

    def R(self,z):
        '''
        R(z): radius of curvature of the wavefronts
        Rz = kz*wm**2 * (cos(z/zm)**2+(sin(z/zm)*zm/zR)**2) / (((zm/zR)**2-1)*sin(2*z/zm))
        '''
        kz = self.kz
        wm = self.wm
        zR = self.zR
        zm = self.zm
        z_zm = z/zm
        zm_zR = zm/zR
        Rz = kz*wm**2 * (cos(z_zm)**2+(sin(z_zm)*zm_zR)**2) / (((zm_zR)**2-1)*sin(2*z_zm))
        return Rz

    def phi_Gouy(self,z):
        '''
        Gouy phase shift
        '''
        n = self.n
        ell = self.ell
        zm = self.zm
        zR = self.zR
        N = 2*n+abs(ell)+1
        theta_Gouy = N * atan((zm/zR)*tan(z/zm)) + ell*z/zm
        return theta_Gouy

    def Psi(self,rho,theta,z):
        '''
        Cylindrical coords
             rho  :  [m]  :  radial coord
            theta : [rad] : azimuth angle
              z   :  [m]  :   axial coord
        '''
        rho,theta,z = asarray(rho),asarray(theta),asarray(z)
        n, ell = self.n, self.ell
        wz=self.w(z)
        r_wz = rho/wz
        kz = self.kz
        A = self.C_nl(n,ell)/wz * ((2**0.5)*(r_wz))**abs(ell) * eval_genlaguerre(n,abs(ell),2*(r_wz)**2) * exp(-(r_wz)**2)
        q = ell*theta + 0.5*kz*(rho**2)/self.R(z) - self.phi_Gouy(z)
        return A * exp(1j*q)

    @staticmethod
    def _test():
        Bz = 1.9 #[T] magnetic field
        vz = 0.7*c #[m/s] velocity of the beam
        w0 = 20e-9 #[m] beam waist
        Psi = LG_nl_beam(n=-1,ell=1,w0=w0,vz=vz,Bz=Bz)
        zm = Psi.zm

        rho = np.linspace(0,8*w0,100)
        theta = np.linspace(0,2*pi,200)
        z = np.linspace(0,zm,100)
        r_mesh,theta_mesh,z_mesh = np.meshgrid(rho,theta,z)

        x_mesh = r_mesh * cos(theta_mesh)
        y_mesh = r_mesh * sin(theta_mesh)
        z_mesh = z_mesh  # z remains the same

        data = Psi(r_mesh,theta_mesh,z_mesh)

        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science','nature','no-latex'])#,'dark_background'])
        from cmfunc import complex_to_rgb, hue_plate

        # figure 1
        _, ax = plt.subplots(1,figsize=(11.26/2.54, 9/2.54), dpi=300)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_xlim(-8*w0, +8*w0)
        ax.set_ylim(-8*w0, +8*w0)
        ax.grid(False)
        z_node = -1
        ax.set_title(r'$|\Psi_{n\ell}|^2(x,y,z=z_m)$')
        img = ax.pcolormesh(x_mesh[:,:,z_node], y_mesh[:,:,z_node], abs(data[:,:,z_node]**2), cmap='bone', shading='gouraud')
        cbar = plt.colorbar(img, ax=ax, ticks=[img.get_array().min(), 0, img.get_array().max()])
        cbar.ax.set_yticklabels(['Min', '0', 'Max'])
        cbar.set_label(r'$|\Psi_{n\ell}|^2$', labelpad=-20, y=1.08, rotation=0, fontsize=10)
        plt.show()

        # figure 2
        _, ax = plt.subplots(1,figsize=(9/2.54, 9/2.54), dpi=300, subplot_kw={'projection':'polar'})
        # ax.set_xlabel(r'$\rho_{\perp}$ [m]')
        # ax.set_ylabel(r'$\theta$ [rad]')
        ax.set_title(r'$\Psi_{n\ell}(\rho,\theta,z=z_m)$')
        z_node = -1
        ax.grid(False)
        ax.axis(False)
        ax.pcolormesh(theta_mesh[:,:,z_node], r_mesh[:,:,z_node],
            complex_to_rgb(data[:,:,z_node]), shading='gouraud'#, interpolation='bilinear'
        )
        plt.show()

        # figure 3
        _, _ = hue_plate()
        plt.show()

        # figure 4
        _, ax = plt.subplots(1,figsize=(11.26/2.54, 9/2.54), dpi=300)
        ax.set_xlabel(r'$z$ [m]')
        ax.set_ylabel(r'$\rho$ [m]')
        ax.set_title(r'$|\Psi_{n\ell}|^2(\rho,\theta=0,z)$')
        ax.set_xlim(0,zm)
        ax.set_ylim(0,8*w0)
        theta_node = 0
        img = ax.pcolormesh(z_mesh[theta_node,:,:], r_mesh[theta_node,:,:],
            abs(data[theta_node,:,:])**2, cmap='bone', shading='gouraud')
        cbar = plt.colorbar(img, ax=ax, ticks=[img.get_array().min(), 0, img.get_array().max()])
        cbar.ax.set_yticklabels(['Min', '0', 'Max'])
        cbar.set_label(r'$|\Psi_{n\ell}|^2$', labelpad=-20, y=1.08, rotation=0, fontsize=10)
        plt.show()

        # figure 5
        _, ax = plt.subplots(1,figsize=(9/2.54, 9/2.54), dpi=300)
        # ax.set_aspect('equal')
        ax.set_xlabel(r'$\rho$ [m]')
        ax.set_ylabel(r'$z$ [m]')
        ax.set_title(r'$\Psi_{n\ell}(\rho,\theta=0,z)$')
        ax.set_xlim(0,zm)
        ax.set_ylim(0,8*w0)
        theta_node = 0
        ax.pcolormesh(z_mesh[theta_node,:,:], r_mesh[theta_node,:,:],
            complex_to_rgb(data[theta_node,:,:]), shading='gouraud')
        plt.show()

        plt.close('all')
        return None

# if __name__ == "__main__":
#     LG_nl_beam._test()

class Bessel_nl_beam(cylindrical_wavefunc):
    def __init__(self,n:int,ell:int,w0:float,vz:float,Bz:float):
        '''
        n: [uint]   radial index
        ell: [int]  topological charge
        w0: [m]     beam waist
        vz: [m/s]   velocity of the beam
        Bz: [T]     magnetic field
        '''

        self.n = abs(int(n))
        self.ell = int(ell)
        self.w0 = float(w0)
        self.vz = float(vz)
        self.Bz = float(Bz)

        kz = m_e*gamma(vz)*vz/hbar #[1/m] wavenumber
        zR = 0.5*kz*w0**2 #[m] Rayleigh distance
        wm = 2*(hbar/abs(e*Bz))**0.5 #[m] magnetic length scale
        zm = 0.5*kz*wm**2 #[m] reduced Larmor distance

        self.kz = kz
        self.zR = zR
        self.wm = wm
        self.zm = zm

    # def __call__(self,rho,theta,z):
    #     return self.Psi(rho,theta,z)

    def Psi(self,rho,theta,z):
        raise NotImplementedError("Bessel beam is not implemented yet.")

class BG_nl_beam(cylindrical_wavefunc):
    def __init__(self,n:int,ell:int,w0:float,vz:float,Bz:float):
        '''
        n: [uint]   radial index
        ell: [int]  topological charge
        w0: [m]     beam waist
        vz: [m/s]   velocity of the beam
        Bz: [T]     magnetic field
        '''

        self.n = abs(int(n))
        self.ell = int(ell)
        self.w0 = float(w0)
        self.vz = float(vz)
        self.Bz = float(Bz)

        kz = m_e*gamma(vz)*vz/hbar #[1/m] wavenumber
        zR = 0.5*kz*w0**2 #[m] Rayleigh distance
        wm = 2*(hbar/abs(e*Bz))**0.5 #[m] magnetic length scale
        zm = 0.5*kz*wm**2 #[m] reduced Larmor distance

        self.kz = kz
        self.zR = zR
        self.wm = wm
        self.zm = zm

    # def __call__(self,rho,theta,z):
    #     return self.Psi(rho,theta,z)

    def Psi(self,rho,theta,z):
        raise NotImplementedError("Bessel-Gauss beam is not implemented yet.")

class LG_nl_packet(cylindrical_wavefunc):
    '''
    Laguerre-Gaussian wave-packet: vortex electrons' wavefunction propagating along an axial uniform magnetic field.

    For definiteness, we choose zero Gouy phase (phi_G = 0) and infinite curvature radius (R = infty), which physically corresponds to a wave-packet initialized at its focus.

    References:
    [1] Karlovets, D. 2019 PRA [10.1103/PhysRevA.99.043824]
    '''
    def __init__(self,n:int,ell:int,wr:float,wz:float,pz:float):
        '''
        n:  [uint]  radial index
        ell:[int]   topological charge
        wr: [m]     packet width
        wz: [m]     packet length
        pz: [N.s]   axial momenta
        '''
        self.n = abs(int(n))
        self.ell = int(ell)
        self.wr = abs(float(wr))
        self.wz = abs(float(wz))
        self.pz = float(pz)

        self._C = Cnl(self.n,self.ell)**0.5

    # def __call__(self,rho,theta,z):
    #     return self.Psi(rho,theta,z)

    def Psi_perp(self,rho,theta):
        n,ell,wr=self.n,self.ell,self.wr
        C = self._C
        # C = (Cnl(n,ell))**0.5
        r_wr = rho/wr
        A = pi**(-1/2) * C * (r_wr)**abs(ell)/wr * eval_genlaguerre(n,abs(ell),r_wr**2) * exp(-0.5*(r_wr)**2)
        q = ell*theta
        return A * exp(1j*q)

    def Psi_para(self,z):
        wz,pz=self.wz,self.pz
        A = pi**(-1/4) * wz**(-1/2) * exp(-0.5*(z/wz)**2)
        q = pz*z/hbar
        return A * exp(1j*q)

    def Psi(self,rho,theta,z):
        return self.Psi_perp(rho,theta)*self.Psi_para(z)

    def visualize(self,rho_mesh,theta_mesh,z_mesh,data):
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science','nature','no-latex'])#,'dark_background'])
        from cmfunc import complex_to_rgb, hue_plate

        wr,wz=self.wr,self.wz

        x_mesh = rho_mesh * cos(theta_mesh)
        y_mesh = rho_mesh * sin(theta_mesh)
        z_mesh = z_mesh  # z remains the same

        # figure 1
        _, ax = plt.subplots(1,figsize=(11.26/2.54, 9/2.54), dpi=300)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_xlim(-8*wr, +8*wr)
        ax.set_ylim(-8*wr, +8*wr)
        ax.grid(False)
        z_node = -1
        ax.set_title(r'$|\Psi_{n\ell}|^2(x,y,z=z_m)$')
        img = ax.pcolormesh(x_mesh[:,:,z_node], y_mesh[:,:,z_node], abs(data[:,:,z_node]**2), cmap='bone', shading='gouraud')
        cbar = plt.colorbar(img, ax=ax, ticks=[img.get_array().min(), 0, img.get_array().max()])
        cbar.ax.set_yticklabels(['Min', '0', 'Max'])
        cbar.set_label(r'$|\Psi_{n\ell}|^2$', labelpad=-20, y=1.08, rotation=0, fontsize=10)
        plt.show()

        # figure 2
        _, ax = plt.subplots(1,figsize=(9/2.54, 9/2.54), dpi=300, subplot_kw={'projection':'polar'})
        # ax.set_xlabel(r'$\rho_{\perp}$ [m]')
        # ax.set_ylabel(r'$\theta$ [rad]')
        ax.set_title(r'$\Psi_{n\ell}(\rho,\theta,z=z_m)$')
        z_node = -1
        ax.grid(False)
        ax.axis(False)
        ax.pcolormesh(theta_mesh[:,:,z_node], rho_mesh[:,:,z_node],
            complex_to_rgb(data[:,:,z_node]), shading='gouraud'#, interpolation='bilinear'
        )
        plt.show()

        # figure 3
        _, _ = hue_plate()
        plt.show()

        # figure 4
        _, ax = plt.subplots(1,figsize=(11.26/2.54, 9/2.54), dpi=300)
        ax.set_xlabel(r'$z$ [m]')
        ax.set_ylabel(r'$\rho$ [m]')
        ax.set_title(r'$|\Psi_{n\ell}|^2(\rho,\theta=0,z)$')
        ax.set_xlim(-8*wz,+8*wz)
        ax.set_ylim(0,8*wr)
        theta_node = 0
        img = ax.pcolormesh(z_mesh[theta_node,:,:], rho_mesh[theta_node,:,:],
            abs(data[theta_node,:,:])**2, cmap='bone', shading='gouraud')
        cbar = plt.colorbar(img, ax=ax, ticks=[img.get_array().min(), 0, img.get_array().max()])
        cbar.ax.set_yticklabels(['Min', '0', 'Max'])
        cbar.set_label(r'$|\Psi_{n\ell}|^2$', labelpad=-20, y=1.08, rotation=0, fontsize=10)
        plt.show()

        # figure 5
        _, ax = plt.subplots(1,figsize=(9/2.54, 9/2.54), dpi=300)
        # ax.set_aspect('equal')
        ax.set_xlabel(r'$\rho$ [m]')
        ax.set_ylabel(r'$z$ [m]')
        ax.set_title(r'$\Psi_{n\ell}(\rho,\theta=0,z)$')
        ax.set_xlim(-8*wz,+8*wz)
        ax.set_ylim(0,8*wr)
        theta_node = 0
        ax.pcolormesh(z_mesh[theta_node,:,:], rho_mesh[theta_node,:,:],
            complex_to_rgb(data[theta_node,:,:]), shading='gouraud')
        plt.show()

        plt.close('all')
        return None

    @staticmethod
    def _test():
        wr = 20e-9 #[m] wave-packet width
        wz = 20e-9 #[m] wave-packet length
        pz = m_e*0.7*c #[N.s] axial momenta
        Psi = LG_nl_packet(n=-1,ell=1,wr=wr,wz=wz,pz=pz)

        rho = np.linspace(0,8*wr,100)
        theta = np.linspace(0,2*pi,200)
        z = np.linspace(-8*wz,+8*wz,200)
        rho_mesh,theta_mesh,z_mesh = np.meshgrid(rho,theta,z)

        data = Psi.Psi(rho_mesh,theta_mesh,z_mesh)

        Psi.visualize(rho_mesh,theta_mesh,z_mesh,data)

        return Psi, data, rho_mesh,theta_mesh,z_mesh

# if __name__ == "__main__":
#     Psi, data, _, _, _ = LG_nl_packet._test()

class Landau_eigenket(cylindrical_wavefunc):
    '''
    Landau eigenket: eigenfunction of single electron in uniform axial magnetic field.
    Note that the axis-symmetrical gauge is opted.

    References:
    [1] Greenshields; Stamps; Franke-Arnold; Barnett. 2014 PRL [10.1103/PhysRevLett.113.240404]
    [2] Greenshields; Franke-Arnold; Stamps. 2015 NJP [10.1088/1367-2630/17/9/093015]
    '''
    def __init__(self,n:int,ell:int,pz:float,wz:float,Bz:float,r0=[0,0,0]):
        '''
        n:  [uint]  radial index
        ell:[int]   topological charge
        pz: [N.s]   axial canonical momenta
        wr: [m]     packet width
        wz: [m]     packet length
        Bz: [T]     axial magnetic field
        r0:=[rho0,theta0,z0]     drifted coordinates with respect to the origin
        '''
        self.n = abs(int(n))
        self.ell = int(ell)
        wm = (2*hbar / abs(e * float(Bz)))**0.5
        self.wr = wm
        self.wz = abs(float(wz))
        self.pz = float(pz)
        self.r0 = asarray(r0)
        if self.r0.shape != (3,):
            raise ValueError("pi must be a 3D vector.")
        rho0,theta0,z0 = self.r0[0],self.r0[1],self.r0[2]
        self.S = lambda rho,theta,z: 0.5*(e*Bz)*rho0*rho*sin(theta0-theta)+pz*(z-z0)

        self._C = Cnl(self.n,self.ell)**0.5

    # def __call__(self,rho,theta,z):
    #     return self.Psi(rho,theta,z)

    def Psi_perp(self,rho,theta):
        n,ell,wr=self.n,self.ell,self.wr
        C = self._C
        # C = (Cnl(n,ell))**0.5
        r_wr = rho/wr
        A = pi**(-1/2) * C * (r_wr)**abs(ell)/wr * eval_genlaguerre(n,abs(ell),r_wr**2) * exp(-0.5*(r_wr)**2)
        q = ell*theta
        return A * exp(1j*q)

    def Psi_para(self,z):
        wz=self.wz
        pz=self.pz
        A = pi**(-1/4) * wz**(-1/2) * exp(-0.5*(z/wz)**2)
        # q = (pz*z)/hbar
        return A #* exp(1j*q)

    def Psi(self,rho,theta,z):
        rho,theta,z = asarray(rho),asarray(theta),asarray(z)
        S =self.S
        return self.Psi_perp(rho,theta)*self.Psi_para(z)*exp(1j*S(rho,theta,z)/hbar)

    def visualize(self,rho_mesh,theta_mesh,z_mesh,data):
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science','nature','no-latex'])#,'dark_background'])
        from cmfunc import complex_to_rgb, hue_plate

        wr,wz=self.wr,self.wz

        x_mesh = rho_mesh * cos(theta_mesh)
        y_mesh = rho_mesh * sin(theta_mesh)
        z_mesh = z_mesh  # z remains the same

        # figure 1
        _, ax = plt.subplots(1,figsize=(11.26/2.54, 9/2.54), dpi=300)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_xlim(-8*wr, +8*wr)
        ax.set_ylim(-8*wr, +8*wr)
        ax.grid(False)
        z_node = -1
        ax.set_title(r'$|\Psi_{n\ell}|^2(x,y,z=z_m)$')
        img = ax.pcolormesh(x_mesh[:,:,z_node], y_mesh[:,:,z_node], abs(data[:,:,z_node]**2), cmap='bone', shading='gouraud')
        cbar = plt.colorbar(img, ax=ax, ticks=[img.get_array().min(), 0, img.get_array().max()])
        cbar.ax.set_yticklabels(['Min', '0', 'Max'])
        cbar.set_label(r'$|\Psi_{n\ell}|^2$', labelpad=-20, y=1.08, rotation=0, fontsize=10)
        plt.show()

        # figure 2
        _, ax = plt.subplots(1,figsize=(9/2.54, 9/2.54), dpi=300, subplot_kw={'projection':'polar'})
        # ax.set_xlabel(r'$\rho_{\perp}$ [m]')
        # ax.set_ylabel(r'$\theta$ [rad]')
        ax.set_title(r'$\Psi_{n\ell}(\rho,\theta,z=z_m)$')
        z_node = -1
        ax.grid(False)
        ax.axis(False)
        ax.pcolormesh(theta_mesh[:,:,z_node], rho_mesh[:,:,z_node],
            complex_to_rgb(data[:,:,z_node]), shading='gouraud'#, interpolation='bilinear'
        )
        plt.show()

        # figure 3
        _, _ = hue_plate()
        plt.show()

        # figure 4
        _, ax = plt.subplots(1,figsize=(11.26/2.54, 9/2.54), dpi=300)
        ax.set_xlabel(r'$z$ [m]')
        ax.set_ylabel(r'$\rho$ [m]')
        ax.set_title(r'$|\Psi_{n\ell}|^2(\rho,\theta=0,z)$')
        ax.set_xlim(-8*wz,+8*wz)
        ax.set_ylim(0,8*wr)
        theta_node = 0
        img = ax.pcolormesh(z_mesh[theta_node,:,:], rho_mesh[theta_node,:,:],
            abs(data[theta_node,:,:])**2, cmap='bone', shading='gouraud')
        cbar = plt.colorbar(img, ax=ax, ticks=[img.get_array().min(), 0, img.get_array().max()])
        cbar.ax.set_yticklabels(['Min', '0', 'Max'])
        cbar.set_label(r'$|\Psi_{n\ell}|^2$', labelpad=-20, y=1.08, rotation=0, fontsize=10)
        plt.show()

        # figure 5
        _, ax = plt.subplots(1,figsize=(9/2.54, 9/2.54), dpi=300)
        # ax.set_aspect('equal')
        ax.set_xlabel(r'$\rho$ [m]')
        ax.set_ylabel(r'$z$ [m]')
        ax.set_title(r'$\Psi_{n\ell}(\rho,\theta=0,z)$')
        ax.set_xlim(-8*wz,+8*wz)
        ax.set_ylim(0,8*wr)
        theta_node = 0
        ax.pcolormesh(z_mesh[theta_node,:,:], rho_mesh[theta_node,:,:],
            complex_to_rgb(data[theta_node,:,:]), shading='gouraud')
        plt.show()

        plt.close('all')
        return None

    @staticmethod
    def _test():
        # wr = 20e-9 #[m] wave-packet width
        Bz = 2.0 #[T] magnetic field
        wz = 20e-9 #[m] wave-packet length
        pz = m_e*0.7*c #[N.s] axial momenta
        Psi = Landau_eigenket(n=-1,ell=1,Bz=Bz,wz=wz,pz=pz,r0=[100*nano,-pi/2,0])
        wr = Psi.wr

        rho = np.linspace(0,8*wr,100)
        theta = np.linspace(0,2*pi,200)
        z = np.linspace(-8*wz,+8*wz,200)
        rho_mesh,theta_mesh,z_mesh = np.meshgrid(rho,theta,z)

        data = Psi.Psi(rho_mesh,theta_mesh,z_mesh)

        Psi.visualize(rho_mesh,theta_mesh,z_mesh,data)

        return Psi, data, rho_mesh,theta_mesh,z_mesh

# if __name__ == "__main__":
#     Psi, data, _, _, _ = Landau_eigenket._test()

class vortex_plane_bispinor(cylindrical_wavefunc):
    '''
    Vortex bispinor: free vortex Dirac electron's wave-function.

    References:
    ------------------------------------------------------------
    plane wave:

    [1] Barnett, Stephen M. 2017 PRL [10.1103/PhysRevLett.118.114802]

    wave-packet:

    [2] van Kruining; Hayrapetyan; Cötte. 2017 PRL [10.1103/PhysRevLett.119.030401]
    [3] Rajabi; Berakdar. 2017 PRA [10.1103/PhysRevA.95.063812] and its erratum [0.1103/PhysRevA.95.063812]
    [4] Fukushima; Shimazaki; Wang. 2020 PRD [10.1103/PhysRevD.102.014045]
    [5] Pavlov; Karlovets. 2023 PRD [10.1103/PhysRevD.109.036017]
    '''
    def __init__(self,a:int,b:int,ell:int,wr:float,wz:float,vz:float,t=0):
        '''
        a: [int]   upper spinor index
        b: [int]   lower spinor index
        ell: [int]  topological charge
        wr: [m]     beam waist
        wz: [m]     packet length
        vz: [m/s]   velocity of the beam
        '''
        self.a = int(a)
        self.b = int(b)
        if not ((self.a == 1 and self.b==0) or (self.a == 0 and self.b==1)):
            raise ValueError("a and b must be 0 or 1, and one of them must be 1.")
        self.ell = int(ell)
        self.wr = abs(float(wr))
        self.wz = abs(float(wz))
        self.vz = float(vz)

        self.p0 = gamma(vz)*m_e*vz #[N.s] initial momentum
        self.Uu = (1 + (m_e * c) / (m_e**2 * c**2 + self.p0**2)**0.5)**0.5
        self.Ud = (1 - (m_e * c) / (m_e**2 * c**2 + self.p0**2)**0.5)**0.5
        self.const_mat = self.Uu * np.stack([+a,+b,0,0],axis=-1) + self.Ud * np.stack([0,0,+a,-b],axis=-1)
        if self.b == 0:
            self.var_mat = 0
        else:
            self.var_mat = self.Ud * (2j*hbar*self.ell) / self.p0 * np.stack([0,0,-b,0],axis=-1)
        self.Norm = 1
        self.t = float(t)

    def __call__(self,t,rho,theta,z):
        return self.Psi(t,rho,theta,z)

    @staticmethod
    def KE(p):
        '''
        Returns the relativistic energy.
        '''
        return c * (m_e**2 * c**2 + p**2)**0.5

    def Psi(self,t,rho,theta,z):
        ell, wr, wz, p0 = self.ell, self.wr, self.wz, self.p0
        E = self.KE(p0)  # [J] relativistic kinetic energy
        # rho, theta, z = asarray(rho), asarray(theta), asarray(z)
        scalar = self.Norm/(2 * wz**0.5 * wr) * exp((p0*z - E*t)/(1j*hbar)) * exp(1j*ell*theta) * (rho/wr)**abs(ell)
        if self.var_mat == 0:
            spinor = self.const_mat
        else:
            spinor = self.const_mat + (exp(-1j*theta)/rho)*self.var_mat
        return np.einsum('...,l->...l', scalar, spinor)

    def visualize(self,t,Lx,Ly,Lz,Nx,Ny,Nz):
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science','nature','no-latex'])#,'dark_background'])
        from cmfunc import complex_to_rgb, hue_plate

        from coords import cartesian_to_cylindrical
        def wavefunc_cartesian(t,x,y,z):
            rho, theta, z = cartesian_to_cylindrical(x,y,z)
            return self.Psi(t, rho, theta, z)

        x = np.linspace(-Lx/2, +Lx/2, Nx, endpoint=False)
        y = np.linspace(-Ly/2, +Ly/2, Ny, endpoint=False)
        z = np.linspace(-Lz/2, +Lz/2, Nz, endpoint=False)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')
        _Psi = wavefunc_cartesian(t=t,x=x_mesh, y=y_mesh, z=z_mesh).astype(np.complex64)

        def scale_match(qty):
            if not isinstance(qty, (int, float)):
                raise TypeError("the input ({qty}) must be a numeric.")
            if 100*milli >= qty >= 0.1*milli:
                prefix = r"m "
                scale = milli
            elif qty >= 0.1*micro:
                prefix = r"\mu "
                scale = micro
            elif qty >= 0.1*nano:
                prefix = r"n "
                scale = nano
            elif qty >= 0.1*pico:
                prefix = r"p "
                scale = pico
            elif qty >= 0.1*femto:
                prefix = r"f "
                scale = femto
            elif qty >= 0.1*atto:
                prefix = r"a "
                scale = atto
            else:
                prefix = r""
                scale = 1
            return prefix, scale

        fig, ax = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(9, 6.4)
        fig.set_dpi(300)
        fig.set_tight_layout(True)
        fig.set_constrained_layout(True)

        x_prefix, x_scale = scale_match(Lx)
        x_unit = x_prefix + r"m"
        y_prefix, y_scale = scale_match(Ly)
        y_unit = y_prefix + r"m"
        z_prefix, z_scale = scale_match(Lz)
        z_unit = z_prefix + r"m"

        # subplots 1 & 2 (X-Y)
        z_node = int(Nz/2)
        psi_arr = _Psi[:,:,z_node,:]
        data = psi_arr[:,:,0]
        rho_arr = prob_dens(psi_arr, type='bispinor')
        X, Y = x_mesh[:,:,z_node]/x_scale, y_mesh[:,:,z_node]/y_scale
        ax[0,0].set_aspect('equal')
        ax[0,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,0].tick_params(axis='both', labelsize=8)
        ax[0,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,0].grid(False)
        ax[0,0].set_title(r'$|\Psi_{\ell}|^2(x,y,z=0)$')
        img1 = ax[0,0].pcolormesh(
            X, Y,
            rho_arr,
            cmap='bone', shading='gouraud')
        cbar1 = plt.colorbar(img1, ax=ax[0,0],
            ticks=[img1.get_array().min(),
                   img1.get_array().max()],
            orientation='horizontal',
            )
        cbar1.ax.set_xticklabels(['Min', 'Max'])
        ax[1,0].set_aspect('equal')
        ax[1,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[1,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[1,0].tick_params(axis='both', labelsize=8)
        ax[1,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[1,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[1,0].grid(False)
        ax[1,0].set_title(r'$\Psi_{\ell}(x,y,z=0)$')
        ax[1,0].pcolormesh(
            X, Y,
            complex_to_rgb(data), shading='gouraud')

        # subplots 3 & 4 (Z-Y)
        x_node = int(Nx/2)
        psi_arr = _Psi[x_node,:,:,:]
        data = psi_arr[:,:,0]
        rho_arr = prob_dens(psi_arr, type='bispinor')
        Z, Y = z_mesh[x_node,:,:]/z_scale, y_mesh[x_node,:,:]/y_scale
        # ax[0,1].set_aspect('equal')
        ax[0,1].set_box_aspect(1)
        ax[0,1].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,1].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,1].tick_params(axis='both', labelsize=8)
        ax[0,1].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,1].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,1].set_title(r'$|\Psi_{\ell}|^2(x=0,y,z)$')
        img2 = ax[0,1].pcolormesh(
            Z, Y,
            rho_arr,
            cmap='bone', shading='gouraud')
        cbar2 = plt.colorbar(img2, ax=ax[0,1],
                            ticks=[img2.get_array().min(),
                                   img2.get_array().max()],
                            orientation='horizontal')
        cbar2.ax.set_xticklabels(['Min', 'Max'])
        # ax[1,1].set_aspect("equal")
        ax[1,1].set_box_aspect(1)
        ax[1,1].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[1,1].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[1,1].tick_params(axis='both', labelsize=8)
        ax[1,1].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[1,1].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[1,1].set_title(r'$\Psi_{\ell}(x=0,y,z)$')
        ax[1,1].pcolormesh(
            Z, Y,
            complex_to_rgb(data), shading='gouraud')

        # subplots 5 & 6 (Z-X)
        y_node = int(Ny/2)
        psi_arr = _Psi[:,y_node,:,:]
        data = psi_arr[:,:,0]
        rho_arr = prob_dens(psi_arr, type='bispinor')
        Z, X = z_mesh[:,y_node,:]/y_scale, x_mesh[:,y_node,:]/x_scale
        # ax[0,2].set_aspect("equal")
        ax[0,2].set_box_aspect(1)
        ax[0,2].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,2].set_ylabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,2].tick_params(axis='both', labelsize=8)
        ax[0,2].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,2].set_ylim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,2].set_title(r'$|\Psi_{\ell}|^2(x,y=0,z)$')
        img3 = ax[0,2].pcolormesh(
            Z, X,
            rho_arr,
            cmap='bone', shading='gouraud')
        cbar3 = plt.colorbar(img3, ax=ax[0,2],
                            ticks=[img3.get_array().min(),
                                   img3.get_array().max()],
                            orientation='horizontal')
        cbar3.ax.set_xticklabels(['Min', 'Max'])
        # ax[1,2].set_aspect("equal")
        ax[1,2].set_box_aspect(1)
        ax[1,2].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[1,2].set_ylabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[1,2].tick_params(axis='both', labelsize=8)
        ax[1,2].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[1,2].set_ylim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[1,2].set_title(r'$\Psi_{\ell}(x,y=0,z)$')
        ax[1,2].pcolormesh(
            Z, X,
            complex_to_rgb(data), shading='gouraud')

        plt.show()
        plt.close()
        return None

    @staticmethod
    def _test(a=1,b=0,ell=1,vz=0.7*c,t=0):
        wr = 20e-9 #[m] wave-packet width
        wz = 20e-9 #[m] wave-packet length
        Psi = vortex_plane_bispinor(a=a,b=b,ell=ell,wr=wr,wz=wz,vz=vz)

        Psi.visualize(t=t,Lx=milli,Ly=milli,Lz=milli,Nx=200,Ny=200,Nz=200)

        return None
#%%
# if __name__ == "__main__":
#     vortex_plane_bispinor._test(a=1,b=0,ell=2,vz=0.1*c)
#%%
# a,b,ell=1,0,1
# vz=0.7*c
# t=0
# wr = 20e-9 #[m] wave-packet width
# wz = 20e-9 #[m] wave-packet length
# Psi = vortex_bispinor(a=a,b=b,ell=ell,wr=wr,wz=wz,vz=vz)

# #%%
# X,Y,Z = np.meshgrid(
#     np.linspace(-1e-6, 1e-6, 5), # x-coordinates
#     np.linspace(-1e-6, 1e-6, 6), # y-coordinates
#     np.linspace(-1e-6, 1e-6, 7), # z-coordinates
#     indexing='ij' # use 'ij' indexing for 3D meshgrid
# )
# Psi_ = Psi(0,X,Y,Z)
# print(Psi_.shape)
# # %%
# # rho = (Psi_.conj().T) @ Psi_
# rho = np.einsum('...i,...i->...', Psi_.conj(), Psi_)
# print(rho.shape)
# print((np.isreal(rho)).all())
# # %%

class vortex_packet_bispinor(cylindrical_wavefunc):
    '''
    Vortex bispinor: free vortex Dirac electron's wave-function.

    References:
    ------------------------------------------------------------
    plane wave:

    [1] Barnett, Stephen M. 2017 PRL [10.1103/PhysRevLett.118.114802]

    wave-packet:

    [2] van Kruining; Hayrapetyan; Cötte. 2017 PRL [10.1103/PhysRevLett.119.030401]
    [3] Rajabi; Berakdar. 2017 PRA [10.1103/PhysRevA.95.063812] and its erratum [0.1103/PhysRevA.95.063812]
    [4] Fukushima; Shimazaki; Wang. 2020 PRD [10.1103/PhysRevD.102.014045]
    [5] Pavlov; Karlovets. 2023 PRD [10.1103/PhysRevD.109.036017]
    '''
    def __init__(self,n:int,ell:int,s:int,p:float,wr:float,wz:float):
        '''
        n: [uint]   radial index, n = 0,1,2,...
        ell: [int]  topological charge, ell = ..., -2, -1, 0, +1, +2, ...
        s: [int]    axial helicity, s = '+1/2' / '-1/2'
        p: [N.s]    axial momentum, p = ]-@@, +@@[
        wr: [m]     beam waist, wr = ]0, +@@[
        wz: [m]     packet length, wz = ]0, +@@[
        t: [s]      time, t = ]-@@, +@@[
        '''
        # self.anti = False
        self.n = abs(int(n))
        self.ell = int(ell)
        if (s+0.5)%1.0 != 0:
            raise ValueError("s must be a non-zero half-integer, but now s = %g" % s)
        if abs(s) > 0.5:
            raise NotImplementedError("only 1/2-spin particles are implemented, namely, s should be -1/2 or +1/2. But now s = %g" % s)
        self.s = float(s)
        self.p = float(p)
        self.wr = abs(float(wr))
        if self.wr <= 0:
            raise ValueError("wr must be positive, but now wr = %g" % self.wr)
        self.wz = abs(float(wz))
        if self.wz <= 0:
            raise ValueError("wz must be positive, but now wz = %g" % self.wz)

        self.Psi_nl = LG_nl_packet(n=self.n,ell=self.ell,wr=self.wr,wz=self.wz,pz=self.p)
        self.Psi_nl1= LG_nl_packet(n=self.n,ell=self.ell+1,wr=self.wr,wz=self.wz,pz=self.p)

        self.I_ = self.I(n=self.n,ell=self.ell,s=self.s,wr=self.wr) # [J] interaction energy
        self.E_ = self.E(n=self.n,ell=self.ell,s=self.s,p=self.p,wr=self.wr,m0=m_e) # [J] total energy
        self.E0 = m_e * c**2

        c0 = (2 * self.E_ * (self.E_ + self.E0))**(-0.5)
        if self.s == +0.5:
            c1 = self.E_ + self.E0
            c2 = 0
            c3 = p * c
            c4 = 1j * self.I_
        elif self.s == -0.5:
            c1 = 0
            c2 = self.E_ + self.E0
            c3 = -1j * self.I_
            c4 = -p * c
        self.c1 = c0 * c1
        self.c2 = c0 * c2
        self.c3 = c0 * c3
        self.c4 = c0 * c4

    def __call__(self,t,rho,theta,z):
        return self.Psi(t,rho,theta,z)

    @staticmethod
    def I(n,ell,s,wr):
        if s == +0.5:
            PE_ = 2**0.5 * hbar * c / wr * (2*n + abs(ell) + ell + 2)**0.5
        elif s == -0.5:
            PE_ = 2**0.5 * hbar * c / wr * (2*n + abs(ell+1) + ell+1)**0.5
        return PE_
    @staticmethod
    def E(n,ell,s,p,wr,m0):
        I0 = 2 * (hbar * c / wr)**2
        if s == +0.5:
            E_ = (I0 * (2*n + abs(ell) + ell + 2) + (p*c)**2 + (m0**2 * c**4))**0.5
        elif s == -0.5:
            E_ = (I0 * (2*n + abs(ell+1) + ell+1) + (p*c)**2 + (m0**2 * c**4))**0.5
        return E_

    def Psi(self,t,rho,theta,z):
        Psi_nl, Psi_nl1 = self.Psi_nl, self.Psi_nl1
        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
        bispinor = np.stack([
            c1 * Psi_nl.Psi(rho,theta,z),
            c2 * Psi_nl1.Psi(rho,theta,z),
            c3 * Psi_nl.Psi(rho,theta,z),
            c4 * Psi_nl1.Psi(rho,theta,z),
            ], axis=-1)
        scalar = exp((self.E_*t)/(1j*hbar))
        return scalar * bispinor

    def visualize(self,t,Lx,Ly,Lz,Nx,Ny,Nz):
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science','nature','no-latex'])#,'dark_background'])
        from cmfunc import complex_to_rgb, hue_plate

        from coords import cartesian_to_cylindrical
        def wavefunc_cartesian(t,x,y,z):
            rho, theta, z = cartesian_to_cylindrical(x,y,z)
            return self.Psi(t, rho, theta, z)

        x = np.linspace(-Lx/2, +Lx/2, Nx, endpoint=False)
        y = np.linspace(-Ly/2, +Ly/2, Ny, endpoint=False)
        z = np.linspace(-Lz/2, +Lz/2, Nz, endpoint=False)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')
        _Psi = wavefunc_cartesian(t=t,x=x_mesh, y=y_mesh, z=z_mesh).astype(np.complex64)

        def scale_match(qty):
            if not isinstance(qty, (int, float)):
                raise TypeError("the input ({qty}) must be a numeric.")
            if 100*milli >= qty >= 0.1*milli:
                prefix = r"m "
                scale = milli
            elif qty >= 0.1*micro:
                prefix = r"\mu "
                scale = micro
            elif qty >= 0.1*nano:
                prefix = r"n "
                scale = nano
            elif qty >= 0.1*pico:
                prefix = r"p "
                scale = pico
            elif qty >= 0.1*femto:
                prefix = r"f "
                scale = femto
            elif qty >= 0.1*atto:
                prefix = r"a "
                scale = atto
            else:
                prefix = r""
                scale = 1
            return prefix, scale

        fig, ax = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(9, 6.4)
        fig.set_dpi(300)
        fig.set_tight_layout(True)
        fig.set_constrained_layout(True)

        x_prefix, x_scale = scale_match(Lx)
        x_unit = x_prefix + r"m"
        y_prefix, y_scale = scale_match(Ly)
        y_unit = y_prefix + r"m"
        z_prefix, z_scale = scale_match(Lz)
        z_unit = z_prefix + r"m"

        # subplots 1 & 2 (X-Y)
        z_node = int(Nz/2)
        psi_arr = _Psi[:,:,z_node,:]
        data = psi_arr[:,:,0]
        rho_arr = prob_dens(psi_arr, type='bispinor')
        X, Y = x_mesh[:,:,z_node]/x_scale, y_mesh[:,:,z_node]/y_scale
        ax[0,0].set_aspect('equal')
        ax[0,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,0].tick_params(axis='both', labelsize=8)
        ax[0,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,0].grid(False)
        ax[0,0].set_title(r'$|\Psi_{\ell}|^2(x,y,z=0)$')
        img1 = ax[0,0].pcolormesh(
            X, Y,
            rho_arr,
            cmap='bone', shading='gouraud')
        cbar1 = plt.colorbar(img1, ax=ax[0,0],
            ticks=[img1.get_array().min(),
                   img1.get_array().max()],
            orientation='horizontal',
            )
        cbar1.ax.set_xticklabels(['Min', 'Max'])
        ax[1,0].set_aspect('equal')
        ax[1,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[1,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[1,0].tick_params(axis='both', labelsize=8)
        ax[1,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[1,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[1,0].grid(False)
        ax[1,0].set_title(r'$\Psi_{\ell}(x,y,z=0)$')
        ax[1,0].pcolormesh(
            X, Y,
            complex_to_rgb(data), shading='gouraud')

        # subplots 3 & 4 (Z-Y)
        x_node = int(Nx/2)
        psi_arr = _Psi[x_node,:,:,:]
        data = psi_arr[:,:,0]
        rho_arr = prob_dens(psi_arr, type='bispinor')
        Z, Y = z_mesh[x_node,:,:]/z_scale, y_mesh[x_node,:,:]/y_scale
        # ax[0,1].set_aspect('equal')
        ax[0,1].set_box_aspect(1)
        ax[0,1].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,1].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,1].tick_params(axis='both', labelsize=8)
        ax[0,1].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,1].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,1].set_title(r'$|\Psi_{\ell}|^2(x=0,y,z)$')
        img2 = ax[0,1].pcolormesh(
            Z, Y,
            rho_arr,
            cmap='bone', shading='gouraud')
        cbar2 = plt.colorbar(img2, ax=ax[0,1],
                            ticks=[img2.get_array().min(),
                                   img2.get_array().max()],
                            orientation='horizontal')
        cbar2.ax.set_xticklabels(['Min', 'Max'])
        # ax[1,1].set_aspect("equal")
        ax[1,1].set_box_aspect(1)
        ax[1,1].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[1,1].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[1,1].tick_params(axis='both', labelsize=8)
        ax[1,1].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[1,1].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[1,1].set_title(r'$\Psi_{\ell}(x=0,y,z)$')
        ax[1,1].pcolormesh(
            Z, Y,
            complex_to_rgb(data), shading='gouraud')

        # subplots 5 & 6 (Z-X)
        y_node = int(Ny/2)
        psi_arr = _Psi[:,y_node,:,:]
        data = psi_arr[:,:,0]
        rho_arr = prob_dens(psi_arr, type='bispinor')
        Z, X = z_mesh[:,y_node,:]/y_scale, x_mesh[:,y_node,:]/x_scale
        # ax[0,2].set_aspect("equal")
        ax[0,2].set_box_aspect(1)
        ax[0,2].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,2].set_ylabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,2].tick_params(axis='both', labelsize=8)
        ax[0,2].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,2].set_ylim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,2].set_title(r'$|\Psi_{\ell}|^2(x,y=0,z)$')
        img3 = ax[0,2].pcolormesh(
            Z, X,
            rho_arr,
            cmap='bone', shading='gouraud')
        cbar3 = plt.colorbar(img3, ax=ax[0,2],
                            ticks=[img3.get_array().min(),
                                   img3.get_array().max()],
                            orientation='horizontal')
        cbar3.ax.set_xticklabels(['Min', 'Max'])
        # ax[1,2].set_aspect("equal")
        ax[1,2].set_box_aspect(1)
        ax[1,2].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[1,2].set_ylabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[1,2].tick_params(axis='both', labelsize=8)
        ax[1,2].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[1,2].set_ylim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[1,2].set_title(r'$\Psi_{\ell}(x,y=0,z)$')
        ax[1,2].pcolormesh(
            Z, X,
            complex_to_rgb(data), shading='gouraud')

        plt.show()
        plt.close()
        return None

    @staticmethod
    def _test():
        n,ell,s,p,B,t=0,+1,+0.5,0.9*m_e*c,2.0,1000*pico#,0.0
        wr = ((2*hbar)/(e*B))**0.5 #[m] wave-packet width
        wz = 20e-9 #[m] wave-packet length
        Psi = vortex_packet_bispinor(n=n,ell=ell,s=s,p=p,wr=wr,wz=wz)

        Psi.visualize(t=t,Lx=200*nano,Ly=200*nano,Lz=200*nano,Nx=200,Ny=200,Nz=200)

        return None

Landau_eigenket_bispinor = vortex_packet_bispinor
# #%%
# if __name__ == "__main__":
#     # %%
#     vortex_packet_bispinor._test()
#     # %%
#     n,ell,s,p,B,t=0,+1,+0.5,0.9*m_e*c,2.0,100*atto
#     wr = ((2*hbar)/(e*B))**0.5 #[m] wave-packet width
#     wz = 20e-9 #[m] wave-packet length
#     Psi = vortex_packet_bispinor(n=n,ell=ell,s=s,p=p,wr=wr,wz=wz)
#     # %%
#     print(Psi(0,wr,0,wz))
#     # %%
#     rho = prob_dens(Psi(0,wr,0,wz),type='bispinor')
#     print(rho)
#     #%%
#     X,Y,Z = np.meshgrid(
#         np.linspace(-1e-6, 1e-6, 5), # x-coordinates
#         np.linspace(-1e-6, 1e-6, 6), # y-coordinates
#         np.linspace(-1e-6, 1e-6, 7), # z-coordinates
#         indexing='ij' # use 'ij' indexing for 3D meshgrid
#     )
#     Psi_ = Psi(0,X,Y,Z)
#     print(Psi_.shape)
#     # %%
#     # rho = (Psi_.conj().T) @ Psi_
#     rho = np.einsum('...i,...i->...', Psi_.conj(), Psi_)
#     print(rho.shape)
#     print((np.isreal(rho)).all())
#     # %%