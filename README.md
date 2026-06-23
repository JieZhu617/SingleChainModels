# SingleChainModels
Python implementations of statistical-mechanics and semi-analytical models for the stretching response of single polymer chains with deformable bonds.

This repository contains code associated with the following works:
1. **"Stretching Response of a Polymer Chain with Deformable Bonds"** ([Paper Link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.218101))
2. **"Elasticity of a Polymer Chain with Deformable Bonds under Fixed Extension and Constant Force"** (Manuscript not yet formally published.) 

## Description

This repository provides numerical implementations for polymer-chain elasticity models with bond stretching and bond-angle deformation. The codes include statistical-mechanics models solved by transfer-matrix methods, as well as semi-analytical models for efficient evaluation of force-extension behavior. 

The statistical-mechanics models are formulated in different ensemble settings:

- **Gibbs ensemble / constant-force ensemble**: the applied force is prescribed. In the isotropic formulation used in the PRL paper, the response depends only on the force magnitude. In the fixed-extension/constant-force comparison, this ensemble is expressed equivalently as a prescribed force along the pulling direction. The main output is the average extension as a function of force.

- **Fixed-extension ensemble**: the chain end-to-end extension along the pulling direction is prescribed. The main output is the average force as a function of extension.

The repository also includes semi-analytical models:

- **deformable freely rotating chain (dFRC) model**: a reduced model that incorporates both bond stretching and bond-angle deformation. It provides an efficient approximation to the full statistical-mechanics calculations.

- **extensible freely jointed chain (eFJC) model**: a simpler semi-analytical model with freely jointed orientations and extensible bonds, used as a reference model for comparison.

Together, these tools provide a unified framework for studying single-chain elasticity with deformable bonds. They combine fixed-extension and constant-force ensemble descriptions, full statistical-mechanics calculations, and semi-analytical approximations to examine how bond stretching and bond-angle deformation affect polymer-chain mechanics, analyze finite-chain and ensemble effects, and generate predictions that can be compared with experimental data when appropriate.

## Citation

If you find this work useful in your research, please cite:

```
@article{zhu2025stretching,
  title={Stretching Response of a Polymer Chain with Deformable Bonds},
  author={Zhu, Jie and Brassart, Laurence},
  journal={Physical Review Letters},
  volume={134},
  number={21},
  pages={218101},
  year={2025},
  publisher={APS}
}
```

For detailed derivations and formulas, please refer to the corresponding papers.

## Required Python Libraries

To run the codes provided in this repository, please ensure you have the following Python libraries installed:

- `numpy`
- `mpmath`
- `scipy`
- `matplotlib`

For improved computational efficiency, you can utilize parallel computation through:

- `multiprocessing`

These libraries can be installed using `pip`:

```
pip install numpy scipy matplotlib
```

The `multiprocessing` module is typically included in the standard Python library.

## Documents

### Transfer Matrix Calculations

- **TM_DeformableBondAngles.py**: Calculate the bond orientation probabilities and force-extension curve in a chain with deformable bond angles.
- **TM_FixedBondAngles.py**: Calculate the bond orientation probabilities and force-extension curve in a chain with fixed bond angles.
- **TM_FreelyJointedBonds.py**: Calculate the bond orientation probabilities and force-extension curve in a chain with freely jointed bonds.

### Semi-analytical Models

- **dFRC_model.py**: Calculate the force-extension curve using the deformable Freely Rotating Chain (dFRC) model.
- **eFJC_model.py**: Calculate the force-extension curve using the extensible Freely Jointed Chain (eFJC) model.

## Contact

For questions, feedback, or potential collaborations, please contact **Jie Zhu** at jiezhu0617@gmail.com.
