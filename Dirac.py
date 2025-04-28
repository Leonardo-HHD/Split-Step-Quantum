# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: Dirac.py
PROJECT: Quantum Electro-Dynamics in External Fields
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.0
CREATED: 2025-04-24
LAST MODIFIED: 2025-04-27

DESCRIPTION:
This script implements the 3-D time-dependent Dirac matrix equation in external electromagnetic fields, using Strange-spillting Fourier pseudo-spectral methods.
"""

#%% Import libraries, functions and constants
import os, sys
sys.path.append(r'D:\MyWindows\MyProjects\VE-SpectralMethod\scr')  # customized modules therein
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import array, asarray, stack, dot, cross, einsum
from numpy import eye, transpose, ones, full
from numpy import real, imag, conj, angle, iscomplexobj, isrealobj
from numpy import abs, sign, inf, pi, exp, log, sin, asin, cos, acos, tan, atan2, sinh, asinh, cosh, acosh, tanh, atanh
from numpy.linalg import norm

from scipy.constants import c, h, hbar, e, m_e, m_p, epsilon_0 as esp_0, mu_0, k as kB, eV, angstrom, milli, micro, nano, pico, femto, atto
from scipy.ndimage import map_coordinates # map_coordinates() is the best one
# from scipy.interpolate import RegularGridInterpolator, interpn

from mkl_fft import fftn, ifftn

from tqdm import tqdm
from time import time

from abc import ABC,abstractmethod
from numpy import broadcast_shapes

#%% class: Force Fields & Potentials
class Field(ABC):
    """
    Field: A class to represent a field in 4D spacetime.
    """
    def __call__(self,t,x,y,z):
        return self.eval(t,x,y,z)
    @abstractmethod
    def eval(self,t,x,y,z):
        pass
    @staticmethod
    def _test():
        pass
class Static_Field(Field):
    """
    Static_Field: A class to represent a possibly space-varying static field.
    """
    def __init__(self,func_xyz:callable):
        self.func = func_xyz
    def eval(self,t,x,y,z):
        x,y,z = np.asarray(x),np.asarray(y),np.asarray(z)
        return self.func(x,y,z)
class Uniform_Field(Field):
    """
    Uniform_Field: A class to represent a possibly time-varying uniform field.
    """
    def __init__(self,func_t:callable):
        self.func = func_t
    def eval(self,t,x,y,z):
        t,x,y,z = np.asarray(t),np.asarray(x),np.asarray(y),np.asarray(z)
        const_0 = np.asarray(self.func(0))
        const_ = self.func(t)
        if isrealobj(const_0):
            const_ = np.array(const_,dtype=np.float32)
        elif iscomplexobj(const_0):
            const_ = np.array(const_,dtype=np.complex64)
        else:
            raise TypeError("Return value of Callable 'func_t'  must be numeric, but now ",type(const_0),").")
        uniform_ = np.zeros((len(t),*broadcast_shapes(x.shape,y.shape,z.shape),*const_0.shape), dtype=const_.dtype)
        t_=0
        for const in const_:
            uniform_[t_,...]=full((*broadcast_shapes(x.shape,y.shape,z.shape), *const_0.shape), const)
            t_+=1
        return uniform_
class Const_Field(Field):
    """
    Const_Field: A class to represent a constant field.
    """
    def __init__(self,const):
        const = np.asarray(const)
        if isrealobj(const):
            self.const = np.array(const,dtype=np.float32)
        elif iscomplexobj(const):
            self.const = np.array(const,dtype=np.complex64)
        else:
            raise TypeError("Parameter 'const' (",type(const),") must be numeric.")
    def eval(self,t,x,y,z):
        t,x,y,z = np.asarray(t),np.asarray(x),np.asarray(y),np.asarray(z)
        return full((*broadcast_shapes(t.shape,x.shape,y.shape,z.shape), *self.const.shape), self.const)
class ST_Field(Field):
    """
    ST_Field: A class to represent a spatio-temproal field.
    """
    def __init__(self,func_txyz:callable):
        self.func = func_txyz
    def eval(self,t,x,y,z):
        t,x,y,z = np.asarray(t),np.asarray(x),np.asarray(y),np.asarray(z)
        return self.func(t,x,y,z)

#%% class: Solver & Utilities
class Solver(ABC):
    @abstractmethod
    def __init__(self,*args,**kwargs):
        pass
    @abstractmethod
    def _matrix_construct(self,*args,**kwargs):
        pass
    @abstractmethod
    def _step(self):
        pass
    @abstractmethod
    def _save(self):
        pass
    @abstractmethod
    def run(self,Nt:int,*args,**kwargs):
        pass
    @abstractmethod
    def visualize(solver, _Psi, _t=None, offscreen=True, *args, **kwargs):
        pass
    @abstractmethod
    def _test(Nt:int,*args,**kwargs):
        pass

    @staticmethod
    def _read_timing(timing_file:str="timing.txt"):
        timing_ = np.loadtxt(timing_file, skiprows=1, usecols=(0, 1))
        idx_, t_ = timing_[:, 0].astype(int), timing_[:, 1]
        timing_hash = dict(zip(idx_, t_))
        return timing_hash, idx_, t_

    @staticmethod
    def _isoduration(Titr,Dt:int=1,*args,**kwargs):
        if not isinstance(Dt, int) or Dt < 1:
            raise ValueError("Dt must be a positive integer.")
        return Titr % Dt == 0

class OS_Dirac(Solver):
    '''
    Operator-Splitting Method (OSM) for the time evolution operator
    OSM is a numerical method to solve the time-dependent Dirac matrix equation

    iħ∂Ψ/∂t = HΨ, Ψ=[Ψ1,Ψ2,Ψ3,Ψ4]ᵀ

    where H is the Hamiltonian matrix operator.
    The Hamiltonian operator is split into linear and non-linear factors, and the time evolution operator is approximated using the Strang splitting.

    The linear sub-problem is solved in momentum space, and the non-linear part is adressed in position space.
    '''
    def __init__(self,
                 Lx:float,Nx:int,Ly:float,Ny:int,Lz:float,Nz:int,
                 dt:float,
                 tosave:callable,save_kwargs:dict,
                 m:float,q:float,
                 Phi:callable, #E:callable
                 A:callable, #B:callable,
                 Psi0:callable,
                 *args,**kwargs):
        self.Lx,self.Nx,self.Ly,self.Ny,self.Lz,self.Nz = Lx,Nx,Ly,Ny,Lz,Nz
        self.dt = dt
        self.tosave = tosave
        self.save_kwargs = save_kwargs
        self.m,self.q = m,q
        self.Phi = Phi#, E
        self.A = A#,B
        self.Psi0 = Psi0

        # Spatial grid
        x = np.linspace(-Lx/2, +Lx/2, Nx, endpoint=False)
        y = np.linspace(-Ly/2, +Ly/2, Ny, endpoint=False)
        z = np.linspace(-Lz/2, +Lz/2, Nz, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        self.x, self.y, self.z = x, y, z
        self.x0,self.y0,self.z0= x[0],y[0],z[0]
        self.dx,self.dy,self.dz=x[1]-x[0],y[1]-y[0],z[1]-z[0]
        self.X, self.Y, self.Z = X, Y, Z

    def _matrix_construct(self,*args,**kwargs):
        X, Y, Z =self.X, self.Y, self.Z
        Lx,Ly,Lz=self.Lx,self.Ly,self.Lz
        Nx,Ny,Nz=self.Nx,self.Ny,self.Nz
        q, m, dt=self.q, self.m, self.dt

        print("OSM solver constructing...")
        start_time = time()

        kx, ky, kz = np.fft.fftfreq(Nx)*(2*pi*Nx/Lx), np.fft.fftfreq(Ny)*(2*pi*Ny/Ly), np.fft.fftfreq(Nz)*(2*pi*Nz/Lz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.KX, self.KY, self.KZ = KX, KY, KZ

        self.t = float(0.0)
        self.Titr = int(0)
        self._Phi = self.Phi(self.t,X,Y,Z).astype(np.float32)
        self._A = self.A(self.t,X,Y,Z).astype(np.float32)
        # self._A2= einsum('...k,...k->...',self._A,self._A)
        self._Psi = (self.Psi0(self.t,X,Y,Z).astype(np.complex64))[...,None]
        # Note: Psi must be bi-spinor with 4 components, and the shape of Psi0 must be (Nx,Ny,Nz,4).

        # Save the grid and initial wavefunction to .npy files
        if True:
            np.savetxt("x.txt", self.x, fmt='%g', header="x[m]")
            np.savetxt("y.txt", self.y, fmt='%g', header="y[m]")
            np.savetxt("z.txt", self.z, fmt='%g', header="z[m]")
            np.save("kx.npy", self.kx)
            np.save("ky.npy", self.ky)
            np.save("kz.npy", self.kz)
            print("x",self.x.shape,", y",self.y.shape,", z",self.z.shape,", kx",kx.shape,", ky",ky.shape,", kz",kz.shape," saved.")
            np.save("0_Psi.npy", self._Psi)
            print("0_Psi",self._Psi.shape," saved.")
            np.save("0_Phi.npy", self._Phi)
            print("0_Phi",self._Phi.shape," saved.")
            np.save("0_A.npy", self._A)
            print("0_A",self._A.shape," saved.")
            np.savetxt("spacing.txt", [[self.Lx, self.Ly, self.Lz, self.Nx, self.Ny, self.Nz]], fmt='%g %g %g %d %d %d', header="Lx[m] Ly[m] Lz[m] Nx Ny Nz")
            print("spacing.txt saved.")
            np.savetxt("timing.txt", [[self.Titr, self.t]], fmt='%d %g', header="idx t[s]")
            print("timing.txt created.")

        # Linear operator in momentum space
        matL1 = np.zeros((Nx,Ny,Nz,4,4),dtype=np.complex64)
        matL1[:,:,:,2,0] = 1j*KZ
        matL1[:,:,:,3,0] = 1j*KX+KY
        matL1[:,:,:,2,1] = 1j*KX-KY
        matL1[:,:,:,3,1] = -1j*KZ
        matL1[:,:,:,0,2] = 1j*KZ
        matL1[:,:,:,1,2] = 1j*KX+KY
        matL1[:,:,:,0,3] = 1j*KX-KY
        matL1[:,:,:,1,3] = -1j*KZ
        matL1 *= -c
        matL2 = np.zeros((Nx,Ny,Nz,4,4),dtype=np.complex64)
        matL2[...,0,0] = +1
        matL2[...,1,1] = +1
        matL2[...,2,2] = -1
        matL2[...,3,3] = -1
        matL2 *= (m*c**2)/(1j*hbar)
        matL = matL1 + matL2
        self.expLdt=np.exp(matL*dt)
        self.expLdt_2=np.exp(matL*dt/2)

        # Non-linear operator in position space
        AX,AY,AZ = self._A[...,0],self._A[...,1],self._A[...,2]
        matN1 = np.zeros((Nx,Ny,Nz,4,4),dtype=np.complex64)
        matN1[:,:,:,2,0] = 1j*AZ
        matN1[:,:,:,3,0] = 1j*AX+AY
        matN1[:,:,:,2,1] = 1j*AX-AY
        matN1[:,:,:,3,1] = -1j*AZ
        matN1[:,:,:,0,2] = 1j*AZ
        matN1[:,:,:,1,2] = 1j*AX+AY
        matN1[:,:,:,0,3] = 1j*AX-AY
        matN1[:,:,:,1,3] = -1j*AZ
        matN1 *= 1j*c*q/hbar
        matN2 = np.zeros((Nx,Ny,Nz,4,4),dtype=np.complex64)
        matN2[:,:,:,0,0] = self._Phi
        matN2[:,:,:,1,1] = self._Phi
        matN2[:,:,:,2,2] = self._Phi
        matN2[:,:,:,3,3] = self._Phi
        matN2 *= 1j*q/hbar
        matN = matN1 + matN2
        self.expNdt = np.exp(matN*dt)
        self.expNdt_2 = np.exp(matN*dt/2)

        print("OSM solver constructed.")
        end_time = time()
        print("Elapsed time: %.2f seconds." % (end_time - start_time))

    # 2-order
    def _step(self):
        Psi1 = self.expNdt * self._Psi
        Psi2 = ifftn(self.expLdt * fftn(Psi1,axes=(0,1,2)),axes=(0,1,2))
        self._Psi = Psi2
        self.Titr += 1
        self.t = self.Titr * self.dt

    def _Diff(self):
        self._Psi = ifftn(self.expLdt * fftn(self._Psi,axes=(0,1,2)),axes=(0,1,2))
        self.Titr += 1
        self.t = self.Titr * self.dt
    # 3-order (Strang's splitting)
    def _head(self):
        self._Psi = ifftn(self.expLdt_2 * fftn(self._Psi,axes=(0,1,2)),axes=(0,1,2))
    def _body(self):
        self._Psi = self.expNdt * self._Psi
    def _tail(self):
        self._Psi = ifftn(self.expLdt_2 * fftn(self._Psi,axes=(0,1,2)),axes=(0,1,2))
        self.Titr += 1
        self.t = self.Titr * self.dt

    def _save(self):
        np.save("%d_Psi.npy" % self.Titr, self._Psi)
        '''Enable the following lines to save the time-independent potentials'''
        # np.save("%d_Phi.npy" % self.Titr, self._Phi)
        # np.save("%d_A.npy" % self.Titr, self._A)
        with open("timing.txt", "a") as f:
            f.write(f"{self.Titr} {self.t}\n")
        # print("At t = ",self.t," sec, Titr = ",self.Titr," saved.")
    def run(self,Nt:int,*args,**kwargs):
        self._matrix_construct()
        solver = self
        print("(Lx,Ly,Lz) = (",solver.Lx,",",solver.Ly,",",solver.Lz,") [m]")
        print("dt = ",solver.dt," [s]")

        print("OSM time-iteration stepping...")
        start_time = time()
        solver._head()
        for Titr in tqdm(range(1,Nt+1), desc="OSM", unit="Titr"):
            solver._body()
            if solver.tosave(Titr,**(solver.save_kwargs)):
                solver._tail()
                solver._save()
                solver._head()
            else:
                solver._Diff()
        # solver._tail()
        end_time = time()
        print("OSM finished.")

        # Print the number of iterations
        print(f"Loop iteration: {Nt} time-steps.")
        print(f"Elapsed time: {(end_time - start_time):.6f} seconds.")

        return None



    @staticmethod
    def visualize(solver, _Psi, _t=None, offscreen=True, *args, **kwargs):
        import matplotlib as mpl
        if offscreen:
            mpl.use('Agg') # for non-interactive backend
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science','nature','no-latex','dark_background'])
        from utils.cmfunc import complex_to_rgb#, hue_plate
        self = solver
        Lx,Ly,Lz = self.Lx,self.Ly,self.Lz
        Nx,Ny,Nz = self.Nx,self.Ny,self.Nz
        x_mesh, y_mesh, z_mesh = self.X, self.Y, self.Z
        dt,dx,dy,dz = self.dt,self.dx,self.dy,self.dz
        if 'n' in kwargs:
            n = int(kwargs['n'])
        else:
            n = None
        if 'ell' in kwargs:
            ell = int(kwargs['ell'])
        else:
            ell = None

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

        def sgn(x):
            sign =""
            if x > 0:
                sign = "+"
            elif x < 0:
                sign = "-"
            return sign

        fig, ax = plt.subplots(nrows=2, ncols=3)
        if isinstance(_t, (int, float)):
            t_prefix, t_scale = scale_match(dt)
            t_unit = t_prefix + r"s"
            suptitle = r"$\Psi_{n\ell}(t=%g\,\mathrm{%s})$" % (float(_t/t_scale),str(t_unit))
            if n is not None and ell is not None:
                suptitle = r"$\Psi_{n=%d}^{\ell=%s%d}(t=%g\,\mathrm{%s})$" % (n,sgn(ell),ell,float(_t/t_scale),str(t_unit))
            fig.suptitle(suptitle, fontsize=10)
            fig.set_size_inches(9, 6.7)
        else:
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
        data = _Psi[:,:,z_node]
        X, Y = x_mesh[:,:,z_node]/x_scale, y_mesh[:,:,z_node]/y_scale
        ax[0,0].set_aspect('equal')
        ax[0,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,0].tick_params(axis='both', labelsize=8)
        ax[0,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,0].grid(False)
        ax[0,0].set_title(r'$|\Psi_{n\ell}|^2(x,y,z=0)$')
        img1 = ax[0,0].pcolormesh(
            X, Y,
            abs(data**2),
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
        ax[1,0].set_title(r'$\Psi_{n\ell}(x,y,z=0)$')
        ax[1,0].pcolormesh(
            X, Y,
            complex_to_rgb(data), shading='gouraud')

        # subplots 3 & 4 (Z-Y)
        x_node = int(Nx/2)
        data = _Psi[x_node,:,:]
        Z, Y = z_mesh[x_node,:,:]/z_scale, y_mesh[x_node,:,:]/y_scale
        # ax[0,1].set_aspect('equal')
        ax[0,1].set_box_aspect(1)
        ax[0,1].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,1].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,1].tick_params(axis='both', labelsize=8)
        ax[0,1].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,1].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,1].set_title(r'$|\Psi_{n\ell}|^2(x=0,y,z)$')
        img2 = ax[0,1].pcolormesh(
            Z, Y,
            abs(data)**2, cmap='bone', shading='gouraud')
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
        ax[1,1].set_title(r'$\Psi_{n\ell}(x=0,y,z)$')
        ax[1,1].pcolormesh(
            Z, Y,
            complex_to_rgb(data), shading='gouraud')

        # subplots 5 & 6 (Z-X)
        y_node = int(Ny/2)
        data = _Psi[:,y_node,:]
        Z, X = z_mesh[:,y_node,:]/y_scale, x_mesh[:,y_node,:]/x_scale
        # ax[0,2].set_aspect("equal")
        ax[0,2].set_box_aspect(1)
        ax[0,2].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,2].set_ylabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,2].tick_params(axis='both', labelsize=8)
        ax[0,2].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,2].set_ylim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,2].set_title(r'$|\Psi_{n\ell}|^2(x,y=0,z)$')
        img3 = ax[0,2].pcolormesh(
            Z, X,
            abs(data)**2, cmap='bone', shading='gouraud')
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
        ax[1,2].set_title(r'$\Psi_{n\ell}(x,y=0,z)$')
        ax[1,2].pcolormesh(
            Z, X,
            complex_to_rgb(data), shading='gouraud')

        '''
        # figure 7
        _, _ = hue_plate()
        plt.show()
        '''

        if not offscreen:
            plt.show()
        # plt.savefig("wavefunction.png", dpi=300, bbox_inches='tight')
        plt.close()
        return fig

    @staticmethod
    def _test(Nt:int,*args,**kwargs):
        pass

#%% Test
if __name__ == "__main__":

    Nt, Dt = 10, 1

    # Generating Fields
    constB = Const_Field([0.0, 2.0, 0.0]) # uniform magnetic field in [T]
    staticA = Static_Field(func_xyz=lambda x,y,z:(0.5*np.cross(constB.eval(0, x, y, z), stack([x, y, z], axis=-1))).astype(np.float32))
    constE = Const_Field([0.0, 0.0, 0.0]) # uniform magnetic field in [V/m]
    staticF = Static_Field(func_xyz=lambda x,y,z:(np.vecdot(constE.eval(0, x, y, z),stack([x,y,z], axis=-1))).astype(np.float32))

    # Generating Bispinor Wavefunction
    n,ell,wr,wz,pz=0,+1,20*nano,20*nano,m_e*c*1e-4
    wavefunc_kwargs = {
        'n':n, 'ell': ell,
        'wr': wr, 'wz': wz, 'pz': pz,
    }
    from utils.wavefunc import LG_nl_packet
    wavefunc_cylinderic = LG_nl_packet(n=n,ell=ell,wr=wr,wz=wz,pz=pz)
    from utils.coords import cartesian_to_cylindrical
    def wavefunc_cartesian(t,x,y,z):
        rho, theta, z = cartesian_to_cylindrical(x,y,z)
        return wavefunc_cylinderic(rho, theta, z)
    def bispinor(t,x,y,z):
        psi = wavefunc_cartesian(t,x,y,z)
        bispinor = np.einsum('ijk,l->ijkl', psi, np.array([1,0,0,0]))
        return bispinor

    # Constructing Solver
    init_kwargs_dict = {
        'Lx':500*nano,'Nx':128,
        'Ly':500*nano,'Ny':128,
        'Lz':500*nano,'Nz':128,
        'dt':0.8*pico,
        'tosave':Solver._isoduration,
        'save_kwargs':{'Dt':Dt},
        'q':-e,'m':m_e,
        'Phi':staticF,
        'A':staticA,
        'Psi0':bispinor,
    }
    solver = OS_Dirac(**init_kwargs_dict)

    # Computating
    simu_kwargs_dict = {'Nt':Nt,'Dt':Dt}
    solver.run(**simu_kwargs_dict)

    # Read data from .TXT files
    timing_hash, Titr_, t_ = solver._read_timing("timing.txt")

    # Visualize the results
    print("Visualize...")
    start_time = time()
    for it in tqdm(range(0,Nt+1,Dt), desc="Visualize", unit="snapshot"):
        _Psi = np.load("%d_Psi.npy"%it)
        _t = timing_hash[it]
        fig = OS_Dirac.visualize(solver,_Psi[...,0,0],_t,n=n,ell=ell,offscreen=True)
        fig.savefig("%d_Psi.jpg"%it, dpi=300, bbox_inches='tight')

    # Create a GIF from the saved images
    from PIL import Image
    # Collect all images and create a GIF
    images = []
    for it in tqdm(range(0,Nt+1,Dt), desc="Animating", unit="snapshot"):
        image_path = f"{it}_Psi.jpg"
        images.append(Image.open(image_path))
    images[0].save('Psi.gif', save_all=True, append_images=images[1:], duration=125, loop=0)
    print("GIF saved as Psi.gif")

    ''' Decomment the following lines to remove the individual images after creating the GIF
    # Clean up the individual images
    for it in range(0, Nt + 1, Dt):
        os.remove(f"{it}_Psi.jpg")'''

    end_time = time()
    print("Elapsed time: %.2f seconds." % (end_time - start_time))
    print("=============== All done. ===============")