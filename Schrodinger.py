# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: Maxwell_Schrodinger.py
PROJECT: Quantum Dynamics in External Fields
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.0
CREATED: 2025-04-16
LAST MODIFIED: 2025-04-16

DESCRIPTION:
This script implements the 3-D time-dependent Schrödinger equation in external electromagnetic fields, using semi-Lagrange Strange-spillting Fourier pseudo-spectral methods.
"""

#%% Import libraries, functions and constants
import os, sys
sys.path.append(r'D:\MyWindows\MyProjects\VE-SpectralMethod\scr')  # customized modules therein
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import array, asarray, stack, dot, cross, einsum
from numpy import eye, transpose, ones, full
I = eye(3) # 3x3 identity matrix
from numpy import real, imag, conj, angle, iscomplexobj, isrealobj
from numpy import abs, sign, inf, pi, exp, log, sin, asin, cos, acos, tan, atan2, sinh, asinh, cosh, acosh, tanh, atanh
from numpy.linalg import norm

from scipy.constants import c, h, hbar, e, m_e, m_p, epsilon_0 as esp_0, mu_0, k as kB, eV, angstrom, milli, micro, nano, pico, femto, atto
from scipy.ndimage import map_coordinates # map_coordinates() is the best one
# from scipy.interpolate import RegularGridInterpolator, interpn

from mkl_fft import fftn, ifftn

from tqdm import tqdm
from time import time

#%% External fields

from numpy import broadcast
def B(t, x, y, z):
    uniformB = np.array([0.0, 2.0, 0.0])  # uniform magnetic field in [T]
    B_ = full((*broadcast(x, y, z).shape, 3), uniformB)
    return B_.astype(np.float32)

@np.vectorize(signature='(),(),(),()->(n)')
def A(t,x,y,z): # vector potential in [T*m]
    x, y, z = asarray(x), asarray(y), asarray(z)
    r = stack([x,y,z], axis=-1) # to make r.shape = (..., 3)
    _B = B(t,x,y,z) # magnetic field in [T]
    # A = -1/2 r x B = 1/2 B x r
    return (0.5*cross(_B,r)).astype(np.float32)

@np.vectorize(signature='(),(),(),()->()')
def Phi(t,x,y,z): # scalar potential in [V]
    x, y, z = asarray(x), asarray(y), asarray(z)
    r = stack([x,y,z], axis=-1) # to make r.shape = (..., 3)
    E = array([0,0,0]) # electric field in [V/m]
    # Phi = r . E
    return dot(r,E).astype(np.float32)
    # np.dot(a,b) require a to be (...,3) when b is (3,)

#%% Solver class

class OSM:
    '''
    Operator-Splitting Method (OSM) for the time evolution operator
    OSM is a numerical method to solve the time-dependent Schrödinger equation

    iħ∂Ψ/∂t = HΨ,

    where H is the Hamiltonian operator.
    The Hamiltonian operator is split into kinetic, convection and potential factors, and the time evolution operator is approximated using the Strang splitting method.

    The kinetic part is applied in momentum space, the convection part is applied in position space using semi-Lagrangian method, and the potential part is applied in position space.
    '''
    def __init__(self,
                 Lx:float,Nx:int,Ly:float,Ny:int,Lz:float,Nz:int,
                 dt:float,
                 tosave:callable,save_kwargs:dict,
                 m:float,q:float,
                 Phi:callable, #E:callable
                 A:callable, B:callable,
                 Psi0:callable):
        self.Lx,self.Nx,self.Ly,self.Ny,self.Lz,self.Nz = Lx,Nx,Ly,Ny,Lz,Nz
        self.dt = dt
        self.tosave = tosave
        self.save_kwargs = save_kwargs
        self.m,self.q = m,q
        self.Phi = Phi#, E
        self.A,self.B = A,B
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
        self.coord_ = np.stack([X, Y, Z], axis=-1) # 3d-coord field: (...,3)

    def _matrix_construct(self):
        X, Y, Z =self.X, self.Y, self.Z
        Lx,Ly,Lz=self.Lx,self.Ly,self.Lz
        Nx,Ny,Nz=self.Nx,self.Ny,self.Nz
        q, m, dt=self.q, self.m, self.dt

        print("OSM solver constructing...")
        start_time = time()

        kx = np.fft.fftfreq(Nx)*(2*pi*Nx/Lx)
        ky = np.fft.fftfreq(Ny)*(2*pi*Ny/Ly)
        kz = np.fft.fftfreq(Nz)*(2*pi*Nz/Lz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = (KX**2 + KY**2 + KZ**2).astype(np.float32)
        self.kx, self.ky, self.kz = kx, ky, kz
        self.KX, self.KY, self.KZ = KX, KY, KZ
        self.K2 = K2

        self.t = float(0.0)
        self.Titr = int(0)
        self._Phi = self.Phi(self.t,X,Y,Z).astype(np.float32)
        self._A = self.A(self.t,X,Y,Z).astype(np.float32)
        self._B = self.B(self.t,X,Y,Z).astype(np.float32)
        self._A2= einsum('...k,...k->...',self._A,self._A)
        self._Psi = self.Psi0(self.t,X,Y,Z).astype(np.complex64)

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

        # Discrete kinetic operator in momentum space
        self._D = exp(-0.5j*dt*hbar/m * K2).astype(np.complex64)
        self._D_2 = exp(-0.25j*dt*hbar/m * K2).astype(np.complex64)

        # Discrete potential operator in position space
        self._U = exp(-1j*(dt/hbar)*(q**2/(2*m)*self._A2 - q*(self._Phi))).astype(np.complex64)

        # Discrete advection operator in position space
        self._R = self.RotMatrix().astype(np.float32) # rotation matrix field

        print("OSM solver constructed.")
        end_time = time()
        print("Elapsed time: %.2f seconds." % (end_time - start_time))

    '''# For time-dependent potential
    # Numerical potential operator in position space
    def _U(self,t):
        Phi_update()
        A_update()
        q,m,dt=self.q,self.m,self.dt
        Phi,A = self._Phi,self._A
        return exp(-1j*(dt/hbar)*(q**2/(2*m)*A**2 - q*Phi))
    '''

    ''' Advection term solved with so-called Characteristics method /
        Semi-Lagrangian method / Lie-Advection method
        Advection method / Transport method /
        Transport by Characteristics method /
        Deformation method / Transportation method / ...
    '''
    def RotMatrix(self):
        """
        为3D网格上的每个点构造旋转矩阵

        参数:
        X, Y, Z: 坐标网格
        q: 电荷
        m: 质量
        dt: 时间步长

        返回:
        _R: 形状为(i,j,k,3,3)的旋转矩阵场

        算法原型:

        1. 计算旋转角
            B0 = (B@B).sqrt()
            b = B/B0
            q_mdt_2 = 0.5*q/m*dt
            # theta = tan(q_mdt_2*B0)*b # exact angle
            theta = q_mdt_2*B

        2. 计算反对称矩阵
            # t_cross_I = np.cross(theta, I)
            t_cross_I = np.array([
                [    0    , theta[2],-theta[1]],
                [-theta[2],    0    , theta[0]],
                [ theta[1],-theta[0],    0    ]])

        3. 计算旋转矩阵
            R = I + 2 / (1+theta@theta) * t_cross_I@(t_cross_I + I)

        用例:
        >>> Xi, Yi, Zi = np.meshgrid(
        ...     np.linspace(-nano, +nano, 129, endpoint=True),
        ...     np.linspace(-nano, +nano, 129, endpoint=True),
        ...     np.linspace(-nano, +nano, 129, endpoint=True),
        ...     indexing='ij',
        ... )
        >>> # Xi, Yi, Zi 是原坐标网格
        >>> input = np.stack([Xi, Yi, Zi], axis=-1) # input.shape: (129,129,129,3)
        >>> R_ = Gyrofield(X=Xi, Y=Yi, Z=Zi, q=-e, m=m_e, dt=femto) # R_.shape: (129,129,129,3,3)
        >>> output = np.einsum('...ij,...j->...i', R_, input) # output.shape: (129,129,129,3)
        >>> Xf, Yf, Zf = output[..., 0], output[..., 1], output[..., 2]
        >>> # Xf, Yf, Zf 是旋转后的坐标
        """
        # 确保输入是NumPy数组
        # X_, Y_, Z_ = asarray(X), asarray(Y), asarray(Z)

        # 计算 B
        # B_ = B(self.t, X_, Y_, Z_)  # 形状: (i,j,k,3)
        B_ = self._B

        # 计算 theta
        theta_ = 0.25 * self.q/self.m * self.dt * B_  # 形状: (i,j,k,3)

        # 计算 t_cross_I（反对称矩阵）
        # t_cross_I_ = _t_cross_I_ufunc(theta_)  # 形状: (i,j,k,3,3)
        t_cross_I_ = np.zeros((*theta_.shape[:-1], 3, 3))
        t_cross_I_[..., 0, 1] = theta_[..., 2]
        t_cross_I_[..., 0, 2] = -theta_[..., 1]
        t_cross_I_[..., 1, 0] = -theta_[..., 2]
        t_cross_I_[..., 1, 2] = theta_[..., 0]
        t_cross_I_[..., 2, 0] = theta_[..., 1]
        t_cross_I_[..., 2, 1] = -theta_[..., 0]

        # 计算 (t_cross_I + I)
        t_cross_I_plus_I_ = t_cross_I_ + I  # 形状: (i,j,k,3,3)

        # 计算 theta @ theta
        t_dot_t_ = (theta_ * theta_).sum(axis=-1)  # 形状: (i,j,k)

        # 计算系数
        factor_ = 2 / (1 + t_dot_t_)  # 形状: (i,j,k)

        # 计算矩阵乘积
        matrix_prod_ = t_cross_I_ @ t_cross_I_plus_I_  # 形状: (i,j,k,3,3)

        # 构造旋转矩阵 R = I + factor * (t_cross_I @ (t_cross_I + I))
        R_ = I + factor_[..., None, None] * matrix_prod_  # 形状: (i,j,k,3,3)

        return R_
    def _C(self, Psi1):
        # pushforward mapping
        coord_ = self.coord_ # 3d-coord field: (...,3)
        R_ = self._R # (3,3)-tensor field: (...,3,3)
        pointi = np.einsum('...ij,...j->...i', R_, coord_) # interpoints field: (...,3)
        # physical coordinates which need interpolation
        Xi, Yi, Zi = pointi[...,0], pointi[...,1], pointi[...,2]
        # convert to grid coordinates
        x0, y0, z0 = self.x0, self.y0, self.z0
        dx, dy, dz = self.dx, self.dy, self.dz
        idx_coords = [(Xi-x0)/dx, (Yi-y0)/dy, (Zi-z0)/dz]
        # displacement interpolation
        Psi2 = map_coordinates(input=Psi1, coordinates=idx_coords, order=1, mode='wrap', cval=0.0)
        # the 'order' of spline is 1, which is linear interpolation.
        # the 'wrap' mode is suitable for periodic boundary condition.
        return Psi2

    # 2-order
    def _step(self):
        Psi1 = self._U * self._Psi
        Psi2 = self._C(Psi1)
        Psi3 = ifftn(self._D * fftn(Psi2))
        self._Psi = Psi3
        self.Titr += 1
        self.t = self.Titr * self.dt

    def _Diff(self):
        self._Psi = ifftn(self._D * fftn(self._Psi))
        self.Titr += 1
        self.t = self.Titr * self.dt
    # 3-order (Strang's splitting)
    def _head(self):
        self._Psi = ifftn(self._D_2 * fftn(self._Psi))
        # self.t += 0.5*self.dt
    def _body(self):
        self._Psi = self._C(self._U * self._Psi)
        # self._Psi = self._U * self._Psi # No advection
    def _tail(self):
        self._Psi = ifftn(self._D_2 * fftn(self._Psi))
        # self.t += 0.5*self.dt
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

    @staticmethod
    def _read_timing(timing_file:str="timing.txt"):
        timing_ = np.loadtxt(timing_file, skiprows=1, usecols=(0, 1))
        idx_, t_ = timing_[:, 0].astype(int), timing_[:, 1]
        timing_hash = dict(zip(idx_, t_))
        return timing_hash, idx_, t_

    @staticmethod
    def _isoduration(Titr,Dt:int=1,*args,**kwargs):
        # if len(args) >= 1:
        #     if isinstance(args[0], int) and args[0] >= 1:
        #         Dt = int(args[0])
        if not isinstance(Dt, int) or Dt < 1:
            raise ValueError("Dt must be a positive integer.")
        # if 'Dt' in kwargs:
        #     if isinstance(kwargs['Dt'], int) and kwargs['Dt'] >= 1:
        #         Dt = int(kwargs['Dt'])
        # if 'Titr' in kwargs:
        #     if isinstance(kwargs['Titr'], int) and kwargs['Titr'] >= 1:
        #         Titr = int(kwargs['Titr'])
        return Titr % Dt == 0

    def run(self,Nt:int):
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

    ''' Parameters estimation:
        ===================================
        N = 2*n+abs(ell)+1
        Nx >= 2 * N * L / wr
        dx <= wr/20
        Lx >= 3000*dx
        dz <= wz/2
        Lz >= 40*wz >= 200*dz
        ==> Lx >= 200*wr, Nx >= 4000,
        T >= 2/(28e9*B)
        dt<= min(0.05/(28e9*B), 0.5*Lz/(Nt*vz))
        ===================================
        B = 2.0 T, wr = 5*nano[m], wz = 5*nano[m], vz = c*1e-5[m/s]
        ==> Lx = 1000*nano[m], dx = 0.25*nano[m], Nx = 4000
        ==> Lz = 200*nano[m], dz = 2.5*nano[m], Nz = 80
        ==> T = 40*pico[s], dt = 1*pico[s], Nt = 40
    '''
    @staticmethod
    def _test(Nt:int=135):
        ''' ortho-axial B-field case:
        Larmor period T = 17.9 ps
        Larmor radius r = 85.2 nm
        ==============================================================
        By = 2.0 T, wr = 20*nano[m], wz = 20*nano[m], vz = c*1e-4[m/s]
        Lx = 500*nano[m], dx = 2.5*nano[m], Nx = 256
        Ly = 500*nano[m], dy = 2.5*nano[m], Ny = 256
        Lz = 500*nano[m], dz = 2.5*nano[m], Nz = 256
        T  = 108*pico[s], dt = 0.8*pico[s], Nt = 135
        '''
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
        # L = 150*max(wr,wz)
        # fL= 28e9 * 2.0 # Larmor frequency in [Hz]
        # v = pz/m_e
        # dt= min(femto, 0.01*fL, 2*L/(v*Nt))
        solver = OSM(
            # Lx=1000*nano,Nx=1024,
            # Ly=1000*nano,Ny=1024,
            Lx=500*nano,Nx=128,
            Ly=500*nano,Ny=128,
            Lz=500*nano,Nz=128,
            dt=0.8*pico,
            tosave=OSM._isoduration,
            save_kwargs={'Dt':1},
            m=m_e,q=-e,
            Phi=Phi, #E=E
            A=A, B=B,
            Psi0=wavefunc_cartesian,
            )
        solver.run(Nt=Nt)

        return solver, wavefunc_kwargs

#%% Test the solver and visualize the results
if __name__ == "__main__":
    Nt, Dt = 135, 1

    # Enact the following lines to run the solver
    solver, wavefunc_kwargs = OSM._test(Nt)
    n,ell=wavefunc_kwargs['n'],wavefunc_kwargs['ell']
    wr,wz,pz=wavefunc_kwargs['wr'],wavefunc_kwargs['wz'],wavefunc_kwargs['pz']

    '''# When needs only __init__() but not run()
    n,ell,wr,wz,pz=0,+1,20*nano,20*nano,m_e*c*1e-4

    from utils.wavefunc import LG_nl_packet
    wavefunc_cylinderic = LG_nl_packet(n=n,ell=ell,wr=wr,wz=wz,pz=pz)
    from utils.coords import cartesian_to_cylindrical
    def wavefunc_cartesian(t,x,y,z):
        rho, theta, z = cartesian_to_cylindrical(x,y,z)
        return wavefunc_cylinderic(rho, theta, z)

    solver = OSM(
        # Lx=1000*nano,Nx=1024,
        # Ly=1000*nano,Ny=1024,
        Lx=500*nano,Nx=256,
        Ly=500*nano,Ny=256,
        Lz=250*nano,Nz=256,
        dt=0.8*pico,
        tosave=OSM._isoduration,
        save_kwargs={'Dt':1},
        m=m_e,q=-e,
        Phi=Phi, #E=E
        A=A, B=B,
        Psi0=wavefunc_cartesian,
        )'''

    # Read data from .TXT files
    timing_hash, Titr_, t_ = solver._read_timing("timing.txt")

    '''
    # Test visualization
    _Psi = np.load("%d_Psi.npy"%0)
    _t = timing_hash[0]
    fig = OSM.visualize(solver,_Psi,_t,offscreen=False,n=0,ell=+1)
    fig.savefig("%d_Psi.jpg"%0, dpi=300, bbox_inches='tight')
    '''

    # Visualize the results
    print("Visualize...")
    start_time = time()
    for it in tqdm(range(0,Nt+1,Dt), desc="Visualize", unit="snapshot"):
        _Psi = np.load("%d_Psi.npy"%it)
        _t = timing_hash[it]
        fig = OSM.visualize(solver,_Psi,_t,n=n,ell=ell,offscreen=True)
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