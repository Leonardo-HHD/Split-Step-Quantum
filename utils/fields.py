# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: fields.py
PROJECT: Quantum Dynamics in External Fields
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.0
CREATED: 2025-04-28
LAST MODIFIED: 2025-05-26

DESCRIPTION:
This script implements the external potential fields.
"""

#%% Import libraries, functions and constants
import numpy as np
from numpy import array, asarray, stack, dot, cross, einsum
from numpy import eye, transpose, ones, full
from numpy import real, imag, conj, angle, iscomplexobj, isrealobj
from numpy import abs, sign, inf, pi, exp, log, sin, asin, cos, acos, tan, atan2, sinh, asinh, cosh, acosh, tanh, atanh
from numpy.linalg import norm

from scipy.constants import c, h, hbar, e, m_e, m_p, epsilon_0 as eps_0, mu_0, k as kB, eV, angstrom, milli, micro, nano, pico, femto, atto

from abc import ABC,abstractmethod
from numpy import broadcast_shapes

# if __name__ == "__main__":
#     from relativisity import gamma
# else:
#     from .relativisity import gamma

#%% abstract class: Fields
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

#%% concrete Fields & Potentials
def generate_const_B(const):
    '''
    params:
    -------
    const: (3,)-shaped real. constant B vector in [T].

    returns:
    -------
    staticA: callable. (...,3)-shaped vector potential in [T.m].
    constB: callable. (...,3)-shaped magnetic field in [T].
    '''
    if (np.asarray(const).ndim > 1) or (np.asarray(const).shape[0] != 3):
        raise ValueError("parameter 'const' must be shaped in (3,).")
    constB = Const_Field(const) # uniform magnetic field in [T]
    staticA = Static_Field(func_xyz=lambda x,y,z:(0.5*np.cross(constB.eval(0, x, y, z), stack([x, y, z], axis=-1))))
    return staticA, constB
def generate_const_E(const):
    '''
    params:
    -------
    const: (3,)-shaped real. constant E vector in [T].

    returns:
    -------
    staticF: callable. (...,1)-shaped scalar potential in [V].
    constE: callable. (...,3)-shaped electric field in [V/m].
    '''
    if (np.asarray(const).ndim > 1) or (np.asarray(const).shape[0] != 3):
        raise ValueError("parameter 'const' must be shaped in (3,).")
    constE = Const_Field(const) # uniform magnetic field in [V/m]
    staticF = Static_Field(func_xyz=lambda x,y,z:(np.vecdot(constE.eval(0, x, y, z),stack([x,y,z], axis=-1))))
    return staticF, constE
def generate_Penning_trap(U0,r0,z0,B0):
    '''
    Penning Trap has a saddle-shaped scalar potential and uniform longitudinal magnetic field.
    U(r,z) = U0*(2*z**2 - r**2)/(2*z0**2 + r0**2)
    B(r,z) = B0
    -------
    params:
    -------
    U0: (1,)-shaped real. scalar potential magnitude in [V].
    r0: (1,)-shaped real. characteristic radius in [m].
    z0: (1,)-shaped real. characteristic length in [m].
    B0: (1,)-shaped real. longitudinal uniform magnetic field in [T].

    returns:
    -------
    staticF: callable. (...,1)-shaped scalar potential in [V].
    staticA: callable. (...,3)-shaped vector potential in [T.m].
    staticE: callable. (...,3)-shaped electric field in [V/m].
    constB: callable. (...,3)-shaped magneitc field in [T].

    Example:
    -------
    # [Brown and Gabrielse. RMP1986]: electronU0=+10.22 V,r0=3.35*1.41 mm,z0=3.35 mm,B0=5.872 T
    '''
    for param in [U0, r0, z0, B0]:
        if (np.asarray(param).ndim == 1) and (np.asarray(param).shape[0] == 1):
            raise ValueError("input '%s' shound be (1,)-shaped." % param)
    factor = U0/(2*z0**2 + r0**2)
    staticF = Static_Field(func_xyz=lambda x,y,z: (factor*(2*z**2 - x**2 -y**2)))
    staticE = Static_Field(func_xyz=lambda x,y,z: (factor*np.stack([2*x,2*y,-4*z], axis=-1)))
    staticA, constB = generate_const_B([0,0,B0])
    return staticF, staticA, staticE, constB
def generate_parabola_potential(U0,wx,wy,wz):
    '''
    Central potential has a ellipse-shaped scalar potential.
    U(x,y,z) = U0*((x/wx)**2 + (y/wy)**2 + (z/wz)**2)
    -------
    params:
    -------
    U0: (1,)-shaped real. scalar potential magnitude in [V].
    wx: (1,)-shaped real. characteristic x-width in [m].
    wy: (1,)-shaped real. characteristic y-width in [m].
    wz: (1,)-shaped real. characteristic z-width in [m].

    returns:
    -------
    staticF: callable. (...,1)-shaped scalar potential in [V].
    constE: callable. (...,3)-shaped electric field in [V/m].
    '''
    for param in [U0, wx, wy, wz]:
        if (np.asarray(param).ndim == 1) and (np.asarray(param).shape[0] == 1):
            raise ValueError("input '%s' shound be (1,)-shaped." % param)
    staticF = Static_Field(func_xyz=lambda x,y,z: (U0*((x/wx)**2 + (y/wy)**2 + (z/wz)**2)))
    staticE = Static_Field(func_xyz=lambda x,y,z: (U0*np.stack([-2*x/wx**2,-2*y/wy**2,-2*z/wz**2], axis=-1)))
    return staticF, staticE
def generate_Coulomb_potential(q,r0=[0,0,0],a=0,n=1):
    '''
    Variable soft-core Coulomb potential.
    V(r) = q/(4*pi*eps_0) * (r**n + a**n)**(-1/n)
    E(r) = q/(4*pi*eps_0) * r**(n-1) * (r**n + a**n)**(-(n+1)/n)
    -------
    params:
    -------
    q: fixed charge at position r0.
    r0: (3,)-shaped real. position of the charge in [m].
    a: modified soft ion potential parameter, a >= 0. If a=0, it is the classic Coulomb potential.
    -------
    returns:
    -------
    staticF: callable. (...,1)-shaped scalar potential in [V].
    staticE: callable. (...,3)-shaped electric field in [V/m].
    -------
    Example:
    -------
    # classic Coulomb potential
    >>> staticF, staticE = generate_Coulomb_potential(+1*e,r0=[+0.1*milli,0,0])
    # soft-core potential (Argon: a=0.6245*(a.u.=0.52918Å) ==> Ei=0.59*(a.u.=27.211eV))
    >>> staticF, staticE = generate_Coulomb_potential(+1*e,r0=[+0.1*milli,0,0],a=0.39*(0.52918e-10)**2)
    '''
    if n < 1 or not isinstance(n, int):
        raise ValueError("parameter 'n' must be an integer and greater than or equal to 1.")
    if (np.asarray(r0).ndim > 1) or (np.asarray(r0).shape[0] != 3):
        raise ValueError("parameter 'r0' must be (3,)-shaped.")
    if (np.asarray(a).ndim > 0) or (np.asarray(a).shape != ()):
        raise ValueError("parameter 'a' must be a scalar.")
    if a < 0:
        raise ValueError("parameter 'a' must be greater than or equal to 0.")
    r = lambda x,y,z: ((x-r0[0])**2 + (y-r0[1])**2 + (z-r0[2])**2)**0.5 # radius saclar
    rv= lambda x,y,z: np.stack([x-r0[0],y-r0[1],z-r0[2]], axis=-1) # radius vector
    ru= lambda x,y,z: rv(x,y,z) / r(x,y,z)[...,None]  # unit radius vector
    _r= lambda x,y,z: (r(x,y,z)**n + a**n)**(-1/n) # modified radius scalar
    r_r3 = lambda x,y,z: (r(x,y,z)**(n-1) * (r(x,y,z)**n + a**n)**(-(n+1)/n))[...,None] * ru(x,y,z) # modified radius vector
    Q = q/(4*pi*eps_0)
    staticF = Static_Field(func_xyz=lambda x,y,z: (Q * _r(x,y,z)))
    staticE = Static_Field(func_xyz=lambda x,y,z: (Q * r_r3(x,y,z)))
    return staticF, staticE
def generate_Coulomb_screening(q,r0=[0,0,0],type='classic',**kwargs):
    '''
    screened Coulomb potential.
    U(r) = Q/(4*pi*eps_0) / abs(r) * exp(-abs(r)/Ls)
    E(r) = Q/r**3 * (1 + r/Ls) * exp(-abs(r)/Ls) * [x-x0,y-y0,z-z0]
    -------
    params:
    -------
    Q: (1,)-shaped real. charge in [C].
    type: str. 'classic', 'Debye', 'Thomas-Fermi', 'alkali', 'Yukawa' or 'Stanton-Murillo[PRE2015]'.

    returns:
    -------
    staticF: callable. (...,1)-shaped scalar potential in [V].
    constE: callable. (...,3)-shaped electric field in [V/m].

    examples:
    -------
    # classic Coulomb potential
    >>> staticF, staticE = generate_Coulomb_screening(+1*e,r0=[+0.1*milli,0,0],type="classic",Ls=1*milli)
    # Debye potential
    >>> staticF, staticE = generate_Coulomb_screening(+1*e,r0=[+0.1*milli,0,0],type="Debye",n_e=1e22)
    '''
    def _r(x,y,z):
        return np.sqrt((x-r0[0])**2 + (y-r0[1])**2 + (z-r0[2])**2)
    def _r2(x,y,z):
        return (x-r0[0])**2 + (y-r0[1])**2 + (z-r0[2])**2
    type_list = [
        'classic', 'Debye', 'Thomas-Fermi', 'alkali', 'Yukawa',
        'Stanton-Murillo[PRE2015]',
        ]
    # classic Coulomb potential
    if type == 'classic':
        Q = q/(4*pi*eps_0)
        Ls = kwargs['Ls'] # Screening wave-length in [m]
    elif type == 'Debye':
        ne = kwargs['Ne'] # electron density in [m^-3]
        Te = kwargs['Te'] # electron temperature in [K]
        LD = (eps_0 * kB * Te / (ne * e**2))**0.5 # Debye length in [m]
        Q = q/(4*pi*eps_0)
        Ls = LD # Debye wave-length in [m]
    elif type == 'Thomas-Fermi':
        raise NotImplementedError("Thomas-Fermi potential is not implemented yet.")
    elif type == 'alkali':
        raise NotImplementedError("alkali potential is not implemented yet.")
    elif type == 'Yukawa':
        raise NotImplementedError("Yukawa potential is not implemented yet.")
    elif type == 'Stanton-Murillo[PRE2015]':
        raise NotImplementedError("Stanton-Murillo potential is not implemented yet.")
    else:
        raise ValueError("parameter 'type' must be one of %s." % type_list)

    staticF = Static_Field(func_xyz=lambda x,y,z: (Q / _r(x,y,z) * exp(-_r(x,y,z)/Ls)))
    staticE = Static_Field(func_xyz=lambda x,y,z: ((Q /_r2(x,y,z)**1.5 * (1+_r(x,y,z)/Ls) * exp(-_r(x,y,z)/Ls))[...,None] * np.stack([x-r0[0],y-r0[1],z-r0[2]], axis=-1)))
    return staticF, staticE
def generate_solenoid(type='Glaser',**kwargs):
    '''
    Solenoid field.
    -------
    params:
    -------
    type: str. 'Glaser' etc.
    kwargs: dict. parameters for the solenoid field.
    -------
    returns:
    -------
    staticA: callable. (...,3)-shaped vector potential in [T.m].
    staticB: callable. (...,3)-shaped magnetic field in [T].
    -------
    Example:
    -------
    # Glaser solenoid field
    >>> staticA, staticB = generate_solenoid(type='Glaser',z0=0*milli,wm=0.5*milli,B0=2.0)
    # z0: centroid shift in [m]
    # wm: magnetic width in [m]
    # B0: magnetic field in [T]
    '''
    type_list = [
        'Glaser',
        ]
    if type == 'Glaser':
        # A_ = 0.5 * B(z) * r * {azimuthal unit vector}
        # B(z) = B0 / (1 + ((z - z0)/wm)**2)
        # nabla_z B(z) = 2 * B0 * (z0 - z) / (wm**2 * (1 + (z - z0)**2 / wm**2)**2)
        # B_ = B(z) * {axial unit vector} - 0.5 * rho * nabla_{z} B(z) * {radial unit vector}
        B0 = kwargs['B0'] # magnetic field in [T]
        z0 = kwargs['z0'] # centroid shift in [m]
        wm = kwargs['wm'] # charac radius in [m]
        B = lambda z: (B0 / (1 + ((z - z0)/wm)**2))
        dzB= lambda z: 2 * B0 * (z0 - z) / (wm * (1 + ((z - z0) / wm)**2))**2
        staticA = Static_Field(func_xyz=lambda x,y,z:(0.5*np.cross(stack([0*x, 0*y, B(z)], axis=-1), stack([x, y, z], axis=-1))))
        staticB = Static_Field(func_xyz=lambda x,y,z:stack([0*x, 0*y, B(z)], axis=-1) - 0.5*stack([x, y, 0*z], axis=-1)*dzB(z)[...,None])
    else:
        raise ValueError("parameter 'type' must be one of %s." % type_list)
    return staticA, staticB

#%% test
if __name__ == "__main__":
    # X,Y,Z=np.meshgrid(
    #     np.linspace(-milli,+milli,11),
    #     np.linspace(-milli,+milli,11),
    #     np.linspace(-milli,+milli,11),
    #     indexing='ij')
    X,Y,Z=np.meshgrid(
        np.linspace(-500*nano,+500*nano,21),
        np.linspace(-500*nano,+500*nano,21),
        np.linspace(-500*nano,+500*nano,21),
        indexing='ij')
    # staticF, staticA, staticE, constB = generate_Penning_trap(-1e5,milli,milli,2.0)
    # staticF, staticE = generate_parabola_potential(1e1,inf,inf,milli)
    # staticF, staticE = generate_Coulomb_screening(+1*e,r0=[0,0,+100*nano],type="Debye",Ne=1e14,Te=2.0*11602)
    staticF, staticE = generate_Coulomb_potential(+1*e,r0=[0,0,+200*nano],a=0.39*(0.52918e-10))
    # staticF, staticE = generate_parabola_potential(U0=+1e1,wx=+inf,wy=+inf,wz=10*micro)
    # staticA, staticB = generate_solenoid(type='Glaser',z0=0.05*micro,wm=0.1*micro,B0=2.5)
    staticA, staticB = generate_const_B([0,0,2.0])

    F_ = staticF(0,X,Y,Z).astype(np.float32)
    E_ = staticE(0,X,Y,Z).astype(np.float32)
    A_ = staticA(0,X,Y,Z).astype(np.float32)
    B_ = staticB(0,X,Y,Z).astype(np.float32)

    xnode,ynode,znode = 5,5,5
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science','nature','no-latex'])

    plt.subplots(2,3,figsize=(9/2.54,7/2.54),dpi=300,constrained_layout=True)
    plt.subplot(231)
    plt.gca().set_aspect('equal')
    plt.quiver(X[:,:,znode],Y[:,:,znode],E_[:,:,znode,0],E_[:,:,znode,1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(232)
    plt.gca().set_aspect('equal')
    plt.quiver(Z[:,ynode,:],X[:,ynode,:],E_[:,ynode,:,2],E_[:,ynode,:,0])
    plt.xlabel('z')
    plt.ylabel('x')
    plt.subplot(233)
    plt.gca().set_aspect('equal')
    plt.quiver(Z[xnode,:,:],Y[xnode,:,:],E_[xnode,:,:,2],E_[xnode,:,:,1])
    plt.xlabel('z')
    plt.ylabel('y')
    plt.subplot(234)
    plt.gca().set_aspect('equal')
    plt.pcolormesh(X[:,:,znode],Y[:,:,znode],F_[:,:,znode])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(235)
    plt.gca().set_aspect('equal')
    plt.pcolormesh(Z[:,ynode,:],X[:,ynode,:],F_[:,ynode,:])
    plt.xlabel('z')
    plt.ylabel('x')
    plt.subplot(236)
    plt.gca().set_aspect('equal')
    plt.pcolormesh(Z[xnode,:,:],Y[xnode,:,:],F_[xnode,:,:])
    plt.xlabel('z')
    plt.ylabel('y')

    plt.subplots(2,3,figsize=(9/2.54,7/2.54),dpi=300,constrained_layout=True)
    plt.subplot(231)
    plt.gca().set_aspect('equal')
    plt.quiver(X[:,:,znode],Y[:,:,znode],B_[:,:,znode,0],B_[:,:,znode,1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(232)
    plt.gca().set_aspect('equal')
    plt.quiver(Z[:,ynode,:],X[:,ynode,:],B_[:,ynode,:,2],B_[:,ynode,:,0])
    plt.xlabel('z')
    plt.ylabel('x')
    plt.subplot(233)
    plt.gca().set_aspect('equal')
    plt.quiver(Z[xnode,:,:],Y[xnode,:,:],B_[xnode,:,:,2],B_[xnode,:,:,1])
    plt.xlabel('z')
    plt.ylabel('y')
    plt.subplot(234)
    plt.gca().set_aspect('equal')
    plt.quiver(X[:,:,znode],Y[:,:,znode],A_[:,:,znode,0],A_[:,:,znode,1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(235)
    plt.gca().set_aspect('equal')
    plt.quiver(Z[:,ynode,:],X[:,ynode,:],A_[:,ynode,:,2],A_[:,ynode,:,0])
    plt.xlabel('z')
    plt.ylabel('x')
    plt.subplot(236)
    plt.gca().set_aspect('equal')
    plt.quiver(Z[xnode,:,:],Y[xnode,:,:],A_[xnode,:,:,2],A_[xnode,:,:,1])
    plt.xlabel('z')
    plt.xlabel('z')
    plt.ylabel('y')

    plt.show()
    plt.close()

