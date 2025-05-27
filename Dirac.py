# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: Dirac.py
PROJECT: Quantum Electro-Dynamics in External Fields
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.1
CREATED: 2025-04-24
LAST MODIFIED: 2025-05-10

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
from numpy import inf, pi, exp, log, sin, asin, cos, acos, tan, atan2, sinh, asinh, cosh, acosh, tanh, atanh
from numpy.linalg import norm

from scipy.constants import c, h, hbar, e, m_e, m_p, epsilon_0 as esp_0, mu_0, k as kB, eV, angstrom, milli, micro, nano, pico, femto, atto, zepto, yocto, ronto, quecto
from scipy.ndimage import map_coordinates # map_coordinates() is the best one
# from scipy.interpolate import RegularGridInterpolator, interpn

from mkl_fft import fftn, ifftn

from tqdm import tqdm
from time import time

from abc import ABC,abstractmethod

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
    def _read_timing(timing_file:str="timing.dat"):
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

        E0 = m*c**2
        if dt > hbar/E0:
            print("Warning: E * dt > hbar ==> numerical instability!")

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
        self._Psi = (self.Psi0(self.t,X,Y,Z).astype(np.complex64))#[...,None]
        # Note: Psi must be bi-spinor with 4 components, and the shape of Psi0 must be (Nx,Ny,Nz,4).
        if np.isnan(self._Phi).any():
            raise ValueError("NaN appears in _Phi")
        if np.isinf(self._Phi).any():
            raise ValueError("Inf appears in _Phi")
        if np.isnan(self._A).any():
            raise ValueError("NaN appears in _A")
        if np.isinf(self._A).any():
            raise ValueError("Inf appears in _A")
        if np.isnan(self._Psi).any():
            raise ValueError("NaN appears in _Psi")
        if np.isinf(self._Psi).any():
            raise ValueError("Inf appears in _Psi")

        # Save the grid and initial wavefunction to .npy files
        if True:
            np.savetxt("x.dat", self.x, fmt='%g', header="x[m]")
            np.savetxt("y.dat", self.y, fmt='%g', header="y[m]")
            np.savetxt("z.dat", self.z, fmt='%g', header="z[m]")
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
            np.savetxt("spacing.dat", [[self.Lx, self.Ly, self.Lz, self.Nx, self.Ny, self.Nz]], fmt='%g %g %g %d %d %d', header="Lx[m] Ly[m] Lz[m] Nx Ny Nz")
            print("spacing.dat saved.")
            np.savetxt("timing.dat", [[self.Titr, self.t]], fmt='%d %g', header="idx t[s]")
            print("timing.dat created.")

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
        matL1 *= -1#c
        matL2 = np.zeros((Nx,Ny,Nz,4,4),dtype=np.complex64)
        matL2[...,0,0] = +1
        matL2[...,1,1] = +1
        matL2[...,2,2] = -1
        matL2[...,3,3] = -1
        matL2 *= (m*c)/(1j*hbar)#(m*c**2)/(1j*hbar)
        L_c = matL1 + matL2
        # self.expLdt=np.exp(c*matL*dt)
        # self.expLdt_2=np.exp(c*matL*dt/2)
        T_c = np.sqrt((m*c/hbar)**2 + KX**2 + KY**2 + KZ**2)
        expLdt = np.cos(T_c*c*dt)[...,None,None] - np.where(T_c==0,0,1j*np.sin(T_c*c*dt)/T_c)[...,None,None] * L_c
        expLdt_2 = np.cos(T_c*c*dt/2)[...,None,None] - np.where(T_c==0,0,1j*np.sin(T_c*c*dt/2)/T_c)[...,None,None] * L_c
        self.expLdt = expLdt.astype(np.complex64)
        self.expLdt_2 = expLdt_2.astype(np.complex64)
        print("Linear Operator constructed:",
              self.expLdt.shape,self.expLdt.dtype)
        if np.isinf(self.expLdt).any():
            raise ValueError("Inf appears in Linear operator matrix")
        if np.isnan(self.expLdt).any():
            raise ValueError("NaN appears in Linear operator matrix")

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
        matN1 *= 1j#*c*q/hbar
        # print("matN1:",matN1.shape)
        # if np.isinf(matN1).any():
        #     raise ValueError("Inf appears")
        # if np.isnan(matN1).any():
        #     raise ValueError("NaN appears")
        matN2 = np.zeros((Nx,Ny,Nz,4,4),dtype=np.complex64)
        A0 = self._Phi/c
        matN2[:,:,:,0,0] = A0
        matN2[:,:,:,1,1] = A0
        matN2[:,:,:,2,2] = A0
        matN2[:,:,:,3,3] = A0
        matN2 *= q/hbar#1j*q/hbar
        # print("matN2:",matN2.shape)
        # if np.isinf(matN2).any():
        #     raise ValueError("Inf appears")
        # if np.isnan(matN2).any():
        #     raise ValueError("NaN appears")
        N_ehc = matN1 + matN2
        # print("matN1 + matN2:",N_ehc.shape)
        if np.isinf(N_ehc).any():
            raise ValueError("Inf appears")
        if np.isnan(N_ehc).any():
            raise ValueError("NaN appears")
        # self.expNdt = np.exp(matN*dt)
        # self.expNdt_2 = np.exp(matN*dt/2)
        Amod = np.sqrt(AX**2 + AY**2 + AZ**2)
        # print("U_ebc:",U_ehc.shape)
        # if np.isnan(U_ehc).any():
        #     raise ValueError("NaN appears")
        # if np.isinf(U_ehc).any():
        #     raise ValueError("Inf appears")
        ehc = c*q/hbar
        expNdt = (
            np.cos(matN2*ehc*dt) - 1j*np.sin(matN2*ehc*dt)
            ) * (
            np.cos(Amod*ehc*dt)[...,None,None] - np.where(Amod==0,0,1j*np.sin(Amod*ehc*dt)/Amod)[...,None,None] * matN1
            )
        # print("expNdt:",expNdt.shape)
        if np.isnan(expNdt).any():
            raise ValueError("NaN appears")
        if np.isinf(expNdt).any():
            raise ValueError("Inf appears")
        expNdt_2 = (
            np.cos(matN2*dt/2) - 1j*np.sin(matN2*dt/2)
            ) * (
            np.cos(Amod*ehc*dt/2)[...,None,None] - np.where(Amod==0,0,1j*np.sin(Amod*ehc*dt/2)/Amod)[...,None,None] * matN1
            )
        # print("expNdt_2:",expNdt_2.shape)
        if np.isnan(expNdt_2).any():
            raise ValueError("NaN appears")
        if np.isinf(expNdt_2).any():
            raise ValueError("Inf appears")
        self.expNdt = expNdt.astype(np.complex64)
        self.expNdt_2 = expNdt_2.astype(np.complex64)
        print("Non-linear Operator constructed:",
              self.expNdt.shape,self.expNdt.dtype)
        if np.isnan(self.expNdt).any():
            raise ValueError("NaN appears")
        if np.isinf(self.expNdt).any():
            raise ValueError("Inf appears")

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
        self._Psi = ifftn(self.expLdt @ fftn(self._Psi,axes=(0,1,2)),axes=(0,1,2))
        self.Titr += 1
        self.t = self.Titr * self.dt
    # 3-order (Strang's splitting)
    def _head(self):
        expLdt_2,_Psi = self.expLdt_2,self._Psi
        # print("expLdt_2:",expLdt_2.shape,expLdt_2.dtype)
        fft_Psi = fftn(_Psi,axes=(0,1,2))
        # print("fft_Psi:",fft_Psi.shape,fft_Psi.dtype)
        prod = einsum('...ij,...i->...j',expLdt_2,fft_Psi)
        # print("prod:",prod.shape,prod.dtype)
        ifft_prod = ifftn(prod,axes=(0,1,2))
        # print("ifft_prod:",ifft_prod.shape,ifft_prod.dtype)
        self._Psi = ifft_prod
        # self._Psi = ifftn(self.expLdt_2 @ fftn(self._Psi,axes=(0,1,2)),axes=(0,1,2))
    def _body(self):
        self._Psi = einsum('...ij,...i->...j',self.expNdt, self._Psi)
    def _tail(self):
        expLdt_2,_Psi = self.expLdt_2,self._Psi
        # print("expLdt_2:",expLdt_2.shape,expLdt_2.dtype)
        fft_Psi = fftn(_Psi,axes=(0,1,2))
        # print("fft_Psi:",fft_Psi.shape,fft_Psi.dtype)
        prod = einsum('...ij,...i->...j',expLdt_2,fft_Psi)
        # print("prod:",prod.shape,prod.dtype)
        ifft_prod = ifftn(prod,axes=(0,1,2))
        # print("ifft_prod:",ifft_prod.shape,ifft_prod.dtype)
        self._Psi = ifft_prod
        # self._Psi = ifftn(self.expLdt_2 @ fftn(self._Psi,axes=(0,1,2)),axes=(0,1,2))
        self.Titr += 1
        self.t = self.Titr * self.dt

    # 4-order (Yoshika's method)
    # Not Implemented

    def _save(self):
        np.save("%d_Psi.npy" % self.Titr, self._Psi)
        '''Enable the following lines to save the time-independent potentials'''
        # np.save("%d_Phi.npy" % self.Titr, self._Phi)
        # np.save("%d_A.npy" % self.Titr, self._A)
        with open("timing.dat", "a") as f:
            f.write(f"{self.Titr} {self.t}\n")
        # print("At t = ",self.t," sec, Titr = ",self.Titr," saved.")
    def run(self,Nt:int,*args,**kwargs):
        self._matrix_construct()
        solver = self
        print("(Lx,Ly,Lz) = (",solver.Lx,",",solver.Ly,",",solver.Lz,") [m]")
        print("dt = ",solver.dt," [s]")

        print("OSM time-iteration stepping...")
        start_time = time()
        # print("original '_Psi'",self._Psi.shape)
        solver._head()
        # print("after excuted '_head()'",self._Psi.shape)
        for Titr in tqdm(range(1,Nt+1), desc="OSM", unit="Titr"):
            solver._body()
            # print("after excuted '_body()'",self._Psi.shape)
            if solver.tosave(Titr,**(solver.save_kwargs)):
                solver._tail()
                # print("after excuted '_tail()'",self._Psi.shape)
                solver._save()
                solver._head()
                # print("after excuted '_head()'",self._Psi.shape)
            else:
                solver._Diff()
                # print("after excuted '_Diff()'",self._Psi.shape)
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
        if 's' in kwargs:
            s = float(kwargs['s'])
        else:
            s = None

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
            elif qty >= 0.1*zepto:#1e-21
                prefix = r"z "
                scale = zepto
            elif qty >= 0.1*yocto:#1e-24
                prefix = r"y "
                scale = yocto
            elif qty >= 0.1*ronto:#1e-27
                prefix = r"r "
                scale = ronto
            elif qty >= 0.01*quecto:#1e-30
                prefix = r"q "
                scale = quecto
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
            suptitle = r"$\Psi_{n{\ell}s}(t=%g\,\mathrm{%s})$" % (float(_t/t_scale),str(t_unit))
            if (n is not None) and (ell is not None) and (s is not None):
                suptitle = r"$\Psi_{n=%d}^{\ell=%s%d,s=%s\frac{1}{2}}(t=%g\,\mathrm{%s})$" % (n,sgn(ell),abs(ell),sgn(s),float(_t/t_scale),str(t_unit))
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

        from utils.wavefunc import prob_dens
        # _Psi = np.squeeze(_Psi,axis=-1) # remove the last dimension
        _rho = prob_dens(_Psi, type='bispinor')
        # subplots 1 & 2 (X-Y)
        z_node = int(Nz/2)
        # psi_arr = _Psi[:,:,z_node,:]
        # print("psi_arr:",psi_arr.shape,psi_arr.dtype)
        # data = psi_arr[:,:,0]
        # print("data:",data.shape,data.dtype)
        rho_arr = _rho[:,:,z_node]
        # print("rho_arr:",rho_arr.shape,rho_arr.dtype)
        # norm = mpl.colors.Normalize(vmin=rho_arr.min(), vmax=rho_arr.max())
        X, Y = x_mesh[:,:,z_node]/x_scale, y_mesh[:,:,z_node]/y_scale
        # ax[0,0].set_aspect('equal')
        ax[0,0].set_box_aspect(1)
        ax[0,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,0].tick_params(axis='both', labelsize=8)
        ax[0,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,0].grid(False)
        ax[0,0].set_title(r'$|\Psi|^2(x,y,z=0)$')
        img1 = ax[0,0].pcolormesh(
            X, Y,
            rho_arr,
            cmap='bone', shading='gouraud',
            # norm=norm,
            )
        cbar1 = plt.colorbar(img1, ax=ax[0,0],
            ticks=[img1.get_array().min(),
                   img1.get_array().max()],
            orientation='horizontal',
            )
        cbar1.ax.set_xticklabels(['Min', 'Max'])
        # ax[1,0].set_aspect('equal')
        ax[1,0].set_box_aspect(1)
        ax[1,0].set_xlabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[1,0].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[1,0].tick_params(axis='both', labelsize=8)
        ax[1,0].set_xlim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[1,0].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[1,0].grid(False)
        ax[1,0].set_title(r'$\Psi(x,y,z=0)$')
        # ax[1,0].pcolormesh(
        #     X, Y,
        #     complex_to_rgb(data), shading='gouraud')
        ax[1,0].pcolormesh(
            X, Y,
            rho_arr,
            cmap='bone', shading='gouraud',
            # norm=norm,
            )

        # subplots 3 & 4 (Z-Y)
        x_node = int(Nx/2)
        # psi_arr = _Psi[x_node,:,:,:]
        # data = psi_arr[:,:,0]
        # rho_arr = prob_dens(psi_arr, type='bispinor')
        rho_arr = _rho[x_node,:,:]
        # norm = mpl.colors.Normalize(vmin=rho_arr.min(), vmax=rho_arr.max())
        Z, Y = z_mesh[x_node,:,:]/z_scale, y_mesh[x_node,:,:]/y_scale
        # ax[0,1].set_aspect('equal')
        ax[0,1].set_box_aspect(1)
        ax[0,1].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,1].set_ylabel(r'$y~\mathrm{[%s]}$'%(y_unit),fontsize=10)
        ax[0,1].tick_params(axis='both', labelsize=8)
        ax[0,1].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,1].set_ylim(-0.5*Ly/y_scale, +0.5*Ly/y_scale)
        ax[0,1].set_title(r'$|\Psi|^2(x=0,y,z)$')
        img2 = ax[0,1].pcolormesh(
            Z, Y,
            rho_arr,
            cmap='bone', shading='gouraud',
            # norm=norm,
            )
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
        ax[1,1].set_title(r'$\Psi(x=0,y,z)$')
        # ax[1,1].pcolormesh(
        #     Z, Y,
        #     complex_to_rgb(data), shading='gouraud')
        ax[1,1].pcolormesh(
            Z, Y,
            rho_arr,
            cmap='bone', shading='gouraud',
            # norm=norm,
            )

        # subplots 5 & 6 (Z-X)
        y_node = int(Ny/2)
        # psi_arr = _Psi[:,y_node,:,:]
        # data = psi_arr[:,:,0]
        # rho_arr = prob_dens(psi_arr, type='bispinor')
        rho_arr = _rho[:,y_node,:]
        # norm = mpl.colors.Normalize(vmin=rho_arr.min(), vmax=rho_arr.max())
        Z, X = z_mesh[:,y_node,:]/y_scale, x_mesh[:,y_node,:]/x_scale
        # ax[0,2].set_aspect("equal")
        ax[0,2].set_box_aspect(1)
        ax[0,2].set_xlabel(r'$z~\mathrm{[%s]}$'%(z_unit),fontsize=10)
        ax[0,2].set_ylabel(r'$x~\mathrm{[%s]}$'%(x_unit),fontsize=10)
        ax[0,2].tick_params(axis='both', labelsize=8)
        ax[0,2].set_xlim(-0.5*Lz/z_scale, +0.5*Lz/z_scale)
        ax[0,2].set_ylim(-0.5*Lx/x_scale, +0.5*Lx/x_scale)
        ax[0,2].set_title(r'$|\Psi|^2(x,y=0,z)$')
        img3 = ax[0,2].pcolormesh(
            Z, X,
            rho_arr,
            cmap='bone', shading='gouraud',
            # norm=norm,
            )
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
        ax[1,2].set_title(r'$\Psi(x,y=0,z)$')
        # ax[1,2].pcolormesh(
        #     Z, X,
        #     complex_to_rgb(data), shading='gouraud')
        ax[1,2].pcolormesh(
            Z, X,
            rho_arr,
            cmap='bone', shading='gouraud',
            # norm=norm,
            )

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
    from utils.fields import generate_parabola_potential, generate_const_B
    staticF, staticE = generate_parabola_potential(U0=+1e1,wx=+inf,wy=+inf,wz=10*micro)
    staticA, constB = generate_const_B(const=[0.0,0.0,2.0]) # uniform magnetic field in [T]
    # constB = Const_Field([0.0, 2.0, 0.0]) # uniform magnetic field in [T]
    # staticA = Static_Field(func_xyz=lambda x,y,z:(0.5*np.cross(constB.eval(0, x, y, z), stack([x, y, z], axis=-1))).astype(np.float32))
    # constE = Const_Field([0.0, 0.0, 0.0]) # uniform magnetic field in [V/m]
    # staticF = Static_Field(func_xyz=lambda x,y,z:(np.vecdot(constE.eval(0, x, y, z),stack([x,y,z], axis=-1))).astype(np.float32))

    # Generating Bispinor Wavefunction
    n,ell,s,p=0,+1,+0.5,m_e*c*1e-4
    wr,wz=20*nano,20*nano
    wavefunc_kwargs = {
        'n':n, 'ell': ell, 's': s, 'p':p,
        'wr': wr, 'wz': wz
    }
    from utils.wavefunc import vortex_packet_bispinor
    wavefunc_cylinderic = vortex_packet_bispinor(n=n,ell=ell,s=s,p=p,wr=wr,wz=wz)
    from utils.coords import cartesian_to_cylindrical
    def wavefunc_cartesian(t,x,y,z):
        rho, theta, z = cartesian_to_cylindrical(x,y,z)
        return wavefunc_cylinderic(t, rho, theta, z)

    # Constructing Solver
    init_kwargs_dict = {
        'Lx':500*nano,'Nx':128,
        'Ly':500*nano,'Ny':128,
        'Lz':500*nano,'Nz':128,
        'dt':1.0*zepto,#0.8*pico,
        'tosave':Solver._isoduration,
        'save_kwargs':{'Dt':Dt},
        'q':-e,'m':m_e,
        'Phi':staticF,
        'A':staticA,
        'Psi0':wavefunc_cartesian,
    }
    solver = OS_Dirac(**init_kwargs_dict)

    # # Test discretization
    # solver._matrix_construct()

    #%% Computating
    simu_kwargs_dict = {'Nt':Nt,'Dt':Dt}
    solver.run(**simu_kwargs_dict)

    # Read data from .TXT files
    timing_hash, Titr_, t_ = solver._read_timing("timing.dat")

    #%% Visualize the results
    print("Visualize...")
    start_time = time()
    for it in tqdm(range(0,Nt+1,Dt), desc="Visualize", unit="snapshot"):
        _Psi = np.load("%d_Psi.npy"%it)
        # print("Loading %d_Psi.npy"%it,_Psi.shape,_Psi.dtype)
        _t = timing_hash[it]
        fig = OS_Dirac.visualize(solver,_Psi,_t,n=n,ell=ell,s=s,offscreen=True)
        fig.savefig("%d_Psi.jpg"%it, dpi=300, bbox_inches='tight')

    #%% Create a GIF from the saved images
    from PIL import Image
    # Collect all images and create a GIF
    images = []
    for it in tqdm(range(0,Nt+1,Dt), desc="Animating", unit="snapshot"):
        image_path = f"{it}_Psi.jpg"
        images.append(Image.open(image_path))
    images[0].save('Psi.gif', save_all=True, append_images=images[1:],
                   duration=250,#125
                   loop=0)
    print("GIF saved as Psi.gif")

    ''' Decomment the following lines to remove the individual images after creating the GIF
    # Clean up the individual images
    for it in range(0, Nt + 1, Dt):
        os.remove(f"{it}_Psi.jpg")'''

    end_time = time()
    print("Elapsed time: %.2f seconds." % (end_time - start_time))
    print("=============== All done. ===============")