import numpy as np
from scipy.constants import c

def gamma(V):
    '''
    Lorentz factor
    V: speed of the frame
    c: speed of light
    gamma = 1/sqrt(1-(v/c)^2)
    gamma = 1 iff v=0
    '''
    return 1/(1-(V/c)**2)**0.5

def lorentz_matrix(v):
    '''
    Lorentz transformation matrix
    v:=(Vx,Vy,Vz) 3d-velocity of the particle
    c: speed of light

    X:=(ct,x,y,z) 4d-position of the particle
    X'=(ct',x',y',z') 4d-position of the particle in the moving frame
    X' = L*X
    '''
    V = sum(np.array(v)**2)**0.5
    if V == 0:
        return np.eye(4)
    g = gamma(V)
    L = np.zeros((4,4))
    L[0,0] = g
    L[0,1:] = -g*v[:]/c
    L[1:,0] = -g*v[:]/c
    L[1:,1:] = np.eye(3) - (g-1)*np.outer(v[:],v[:])/V**2
    return L

def STtovec4(t,x,y,z):
    return np.array([c*t,x,y,z])

def vec4toST(X):
    if len(X) != 4:
        raise ValueError("v must be a 4-vector")
    return X[0]/c, X[1], X[2], X[3]

def lorentz_trans(t:float,r:np.ndarray,v:np.ndarray)-> list[float]:
    '''
    Lorentz transformation
    t: time of the particle in the rest frame
    x:=(X,Y,Z) 3d-position of the particle
    v:=(Vx,Vy,Vz) 3d-velocity of the particle

    t': time of the particle in the moving frame
    x':=(X',Y',Z') 3d-position of the particle in the moving frame
    '''
    X0 = STtovec4(t,r[0],r[1],r[2])
    L = lorentz_matrix(v)
    X1 = L @ X0
    t_, x_, y_, z_ = vec4toST(X1)
    return t_, x_, y_, z_

if __name__ == "__main__":
    t = 0.1 # [s]
    r = np.array([1,2,3]) # [m]
    print(f"t = {t} [s]")
    print(f"r = {r} [m]")
    v = np.array([0.5,0.5,0.5])*c # [m/s]
    print(f"v = {v/c} [c]")
    print("Lorentz transformation:")
    t_, x_, y_, z_ = lorentz_trans(t=t, r=r, v=v)
    r_=np.array([x_,y_,z_])
    print(f"t' = {t_} [s]")
    print(f"r' = {r_} [m]")
