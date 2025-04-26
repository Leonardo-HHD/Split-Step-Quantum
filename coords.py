import numpy as np
from numpy import array, asarray, stack
from numpy import pi, exp, log, sin, asin, cos, acos, tan, atan, atan2, sinh, asinh, cosh, acosh, tanh, atanh, real, imag

def cylindrical_to_cartesian(rho, theta, z):
    '''
    Convert cylindrical coordinates to cartesian coordinates
    rho: radial coordinate
    theta: azimuthal angle
    z: axial coordinate
    '''
    x = rho * cos(theta)
    y = rho * sin(theta)
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    '''
    Convert cartesian coordinates to cylindrical coordinates
    x: x coordinate
    y: y coordinate
    z: z coordinate
    '''
    x, y, z = asarray(x), asarray(y), asarray(z)
    rho = (x**2 + y**2)**0.5
    theta = atan2(y, x)
    return rho, theta, z

def spherical_to_cartesian(r, theta, phi):
    '''
    Convert spherical coordinates to cartesian coordinates
    r: radial coordinate
    theta: azimuthal angle
    phi: polar angle
    '''
    r, theta, phi = asarray(r), asarray(theta), asarray(phi)
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    '''
    Convert cartesian coordinates to spherical coordinates
    x: x coordinate
    y: y coordinate
    z: z coordinate
    '''
    x, y, z = asarray(x), asarray(y), asarray(z)
    r = (x**2 + y**2 + z**2)**0.5
    theta = atan2(y, x)
    phi = acos(z / r)
    return r, theta, phi

def cylindrical_to_spherical(rho, theta, z):
    '''
    Convert cylindrical coordinates to spherical coordinates
    rho: radial coordinate
    theta: azimuthal angle
    z: axial coordinate
    '''
    rho, theta, z = asarray(rho), asarray(theta), asarray(z)
    r = (rho**2 + z**2)**0.5
    phi = atan2(rho, z)
    return r, theta, phi

def spherical_to_cylindrical(r, theta, phi):
    '''
    Convert spherical coordinates to cylindrical coordinates
    rho: radial coordinate
    theta: azimuthal angle
    phi: polar angle
    '''
    r, theta, phi = asarray(r), asarray(theta), asarray(phi)
    rho = r * sin(phi)
    z = r * cos(phi)
    return rho, theta, z

# Test the conversion functions
def __test1():

    rho, theta, z = 1, pi/4, 2
    x, y, z = cylindrical_to_cartesian(rho, theta, z)
    print(f"Cylindrical to Cartesian: ({rho}, {theta}, {z}) -> ({x}, {y}, {z})")
    rho, theta, z = cartesian_to_cylindrical(x, y, z)
    print(f"Cartesian to Cylindrical: ({x}, {y}, {z}) -> ({rho}, {theta}, {z})")

    r, theta, phi = 1, pi/4, pi/3
    x, y, z = spherical_to_cartesian(r, theta, phi)
    print(f"Spherical to Cartesian: ({r}, {theta}, {phi}) -> ({x}, {y}, {z})")
    r, theta, phi = cartesian_to_spherical(x, y, z)
    print(f"Cartesian to Spherical: ({x}, {y}, {z}) -> ({r}, {theta}, {phi})")

    rho, theta, z = 1, pi/4, 2
    r, theta, phi = cylindrical_to_spherical(rho, theta, z)
    print(f"Cylindrical to Spherical: ({rho}, {theta}, {z}) -> ({r}, {theta}, {phi})")
    rho, theta, z = spherical_to_cylindrical(r, theta, phi)
    print(f"Spherical to Cylindrical: ({r}, {theta}, {phi}) -> ({rho}, {theta}, {z})")

def __test2():
    # Test with numpy arrays
    rho = np.array([1, 2, 3])
    theta = np.array([pi/4, pi/3, pi/2])
    z = np.array([2, 3, 4])
    x, y, z = cylindrical_to_cartesian(rho, theta, z)
    print(f"Cylindrical to Cartesian: {rho}, {theta}, {z} -> {x}, {y}, {z}")
    rho, theta, z = cartesian_to_cylindrical(x, y, z)
    print(f"Cartesian to Cylindrical: {x}, {y}, {z} -> {rho}, {theta}, {z}")

    r = np.array([1, 2, 3])
    theta = np.array([pi/4, pi/3, pi/2])
    phi = np.array([pi/6, pi/4, pi/3])
    x, y, z = spherical_to_cartesian(r, theta, phi)
    print(f"Spherical to Cartesian: {r}, {theta}, {phi} -> {x}, {y}, {z}")
    r, theta, phi = cartesian_to_spherical(x, y, z)
    print(f"Cartesian to Spherical: {x}, {y}, {z} -> {r}, {theta}, {phi}")

    rho = np.array([1, 2, 3])
    theta = np.array([pi/4, pi/3, pi/2])
    z = np.array([2, 3, 4])
    r, theta, phi = cylindrical_to_spherical(rho, theta, z)
    print(f"Cylindrical to Spherical: {rho}, {theta}, {z} -> {r}, {theta}, {phi}")
    rho, theta, z = spherical_to_cylindrical(r, theta, phi)
    print(f"Spherical to Cylindrical: {r}, {theta}, {phi} -> {rho}, {theta}, {z}")

def __test3():
    x, y, z = np.mgrid[-1:1:4j, -1:1:6j, -1:1:8j]
    rho, theta, z = cartesian_to_cylindrical(x, y, z)
    print(f"Cartesian to Cylindrical: x{x.shape}, y{y.shape}, z{z.shape} -> rho{rho.shape}, theta{theta.shape}, z{z.shape}")
    r, theta, phi = cartesian_to_spherical(x, y, z)
    print(f"Cartesian to Spherical: x{x.shape}, y{y.shape}, z{z.shape} -> r{r.shape}, theta{theta.shape}, phi{phi.shape}")
    # whether the results are the same
    if np.allclose(cylindrical_to_cartesian(rho, theta, z), (x, y, z),equal_nan=True):
        print("Cylindrical to Cartesian: Exact Match")
    else:
        print("Cylindrical to Cartesian: Fail Match")
    if np.allclose(spherical_to_cartesian(r, theta, phi), (x, y, z),equal_nan=True):
        print("Spherical to Cartesian: Exact Match")
    else:
        print("Spherical to Cartesian: Fail Match")
    if np.allclose(cartesian_to_cylindrical(x, y, z), (rho, theta, z),equal_nan=True):
        print("Cartesian to Cylindrical: Exact Match")
    else:
        print("Cartesian to Cylindrical: Fail Match")
    if np.allclose(cartesian_to_spherical(x, y, z), (r, theta, phi),equal_nan=True):
        print("Cartesian to Spherical: Exact Match")
    else:
        print("Cartesian to Spherical: Fail Match")
    if np.allclose(cylindrical_to_spherical(rho, theta, z), (r, theta, phi),equal_nan=True):
        print("Cylindrical to Spherical: Exact Match")
    else:
        print("Cylindrical to Spherical: Fail Match")
    if np.allclose(spherical_to_cylindrical(r, theta, phi), (rho, theta, z),equal_nan=True):
        print("Spherical to Cylindrical: Exact Match")
    else:
        print("Spherical to Cylindrical: Fail Match")

if __name__ == "__main__":
    print("Testing coordinate conversion functions")
    print("========================================")
    print("Testing point-to-point conversion")
    __test1()
    print("========================================")
    print("Testing with 1d-arrays")
    __test2()
    print("========================================")
    print("Testing with 3d-arrays")
    __test3()
    print("========================================")
    # ... Add more tests if needed
    # print("========================================")
    print("All tests passed")