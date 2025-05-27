# Split-Step-Quantum

Operator-Splitting Solvers for the Time-Dependent Schrödinger and Dirac Equations.

## Implementation on the Schrödinger equation

The code is benchmarked with a vortex state electron 3D wave-packet propagating in a uniform magnetic field. The initial wave-front is a Laguerre-Gaussian packet.

---
![Vortex packet](https://github.com/Leonardo-HHD/Split-Step-Quantum/blob/dev/examples/Schrodinger/para_vz%3D0_Bz%3D2.0T/Psi.gif)

This one is the case of:
- initial scales: characteristic radius of 20 nanometers, characteristic length of 20 nanometers
- initial speed: 0
- magnetic field: 2.0 Tesla, along $z$-axis
- scalar potential: none
- mesh grid: regular Cartesian mesh of 128x128x128

This case present the Landau-type eigen-state of vortex electron in uniform magneitc field.

---
![Vortex packet](https://github.com/Leonardo-HHD/Split-Step-Quantum/blob/dev/examples/Schrodinger/para_central_trap_vz%3D0_U0%3D%2B10V_wz%3D10um_Bz%3D2.0T/Psi.gif)

This is the case of:
- initial scales: characteristic radius of 20 nanometers, characteristic length of 20 nanometers
- initial speed: 0
- magnetic field: 2.0 Tesla, along $z$-axis
- scalar potential: parabola central potential along y-dimention, $U(z)=10\mathrm{V}\cdot(\frac{z}{10\mathrm{\mu m}})^2$
- mesh grid: regular Cartesian mesh of 128x128x128

---
![Vortex packet](https://github.com/Leonardo-HHD/Split-Step-Quantum/blob/dev/examples/Schrodinger/perp_central_trap_vz0%3D0_U0%3D%2B10V_wy%3D10um_By%3D2.0T/Psi.gif)

This is the case of:
- initial scales: characteristic radius of 20 nanometers, characteristic length of 20 nanometers
- initial speed: 0
- magnetic field: 2.0 Tesla, along $y$-axis
- scalar potential: parabola central potential along y-dimention, $U(y)=10\mathrm{V}\cdot(\frac{y}{10\mathrm{\mu m}})^2$
- mesh grid: regular Cartesian mesh of 128x128x128

---
![Vortex packet](https://github.com/Leonardo-HHD/Split-Step-Quantum/blob/dev/examples/Schrodinger/perp_central_trap_vz0%3D30kms_U0%3D%2B10V_wy%3D10um_By%3D2.0T/Psi.gif)

This is the case of:
- initial scales: characteristic radius of 20 nanometers, characteristic length of 20 nanometers
- initial speed: 0.01% light-speed, along $z$-axis
- magnetic field: 2.0 Tesla, along $y$-axis
- scalar potential: parabola central potential along y-dimention, $U(y)=10\mathrm{V}\cdot(\frac{y}{10\mathrm{\mu m}})^2$
- mesh grid: regular Cartesian mesh of 256x256x256

This showcases the Larmor gyration of momentum and the gyration of intrinsic orbital angular momentum (iOAM). Here, the gyro-period of iOAM is exact half of the period of Larmor, which is well consistent with the analytical results in [this PRA](https://link.aps.org/doi/10.1103/PhysRevA.86.012701).

## Implementation on the Dirac equation
