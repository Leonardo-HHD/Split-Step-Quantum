# Split-Step-Quantum

Operator-Splitting Solvers for the Time-Dependent Schrödinger and Dirac Equations.

## Implementation on the Schrödinger equation

The code is benchmarked with a vortex state electron 3D wave-packet propagating in a uniform magnetic field. The initial wave-front is a Laguerre-Gaussian packet.

---
![Vortex packet](https://github.com/Leonardo-HHD/Split-Step-Quantum/blob/dev/examples/Schrodinger/vortex_co-axis_vz=0_Bz=2.0T/Psi.gif)

This one is the case of:
- initial beam waist of 0.5 micrometer
- zero-initial momentum
- longitudinal magnetic field of 2.0 Tesla

This case present the Landau-type eigen-state of vortex electron in uniform magneitc field.

---
![Vortex packet](https://github.com/Leonardo-HHD/Split-Step-Quantum/blob/dev/examples/Schrodinger/vortex_ortho_vz0=40kms_By=2.0T/Psi.gif)

This is the case of:
- initial beam waist of 0.5 micrometer
- initial momentum of $0.7m_e c$
- perpendicular magnetic field of 2.0 Tesla

This showcases the Larmor gyration of momentum and the gyration of intrinsic orbital angular momentum (iOAM). Here, the gyro-period of iOAM is exact half of the period of Larmor, which is well consistent with the analytical results in [this PRA](https://link.aps.org/doi/10.1103/PhysRevA.86.012701).

## Implementation on the Dirac equation
