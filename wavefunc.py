import numpy as np
from numpy import array, asarray, stack
from numpy import pi, exp, log, sin, cos, tan, atan, sinh, cosh, tanh, atanh, real, imag
from scipy.constants import c,h,hbar,e,m_e,m_p,epsilon_0,mu_0,k,eV,angstrom
from scipy.special import hermite, factorial, eval_genlaguerre

if __name__ == "__main__":
    from relativisity import gamma
else:
    from .relativisity import gamma


def Cnl(n, ell):
    '''
    Computes the normalization constant for the Laguerre-Gaussian wave-packet.
    '''
    return (factorial(n) / (factorial(n + abs(ell))))

from abc import ABC,abstractmethod
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

class LG_nl_beam(cylindrical_wavefunc):
    '''
    Laguerre-Gaussian beam: vortex electrons' wavefunction propagating along an axial uniform magnetic field.
    References:
    [1] Zou, L.-P. and Zhang, P.-M. and Silenko, A. J. 2021 PRA [https://doi.org/10.1103/PhysRevA.103.L010201]
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
    [1] Karlovets, D. 2019 PRA [https://doi.org/10.1103/PhysRevA.99.043824]
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

if __name__ == "__main__":
    Psi, data, _, _, _ = LG_nl_packet._test()