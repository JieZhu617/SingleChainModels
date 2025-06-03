# SingleChainModels
**Official implementation of the paper:**
**"Stretching Response of a Polymer Chain with Deformable Bonds"** ([Paper Link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.218101))

## Description

This repository contains Python implementations of statistical mechanics and semi-analytical models to study the stretching behavior of polymer chains. Specifically, it includes models accounting for both bond stretching and bond angle deformation, providing accurate predictions of polymer chain responses across the entire force ranges.

The main features include:

- **Statistical Mechanics Models**: Implementation of transfer matrix methods to calculate bond orientation probabilities and force-extension relationships.
- **Semi-Analytical Models**: Efficient computation using deformable Freely Rotating Chain (dFRC) and extensible Freely Jointed Chain (eFJC) models, ideal for rapid evaluation and analysis.

These tools enable researchers to explore polymer chain mechanics comprehensively, offering predictions validated by experimental data without the necessity of parameter fitting.

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

For detailed derivations and formulas, please refer to the full text of the paper.

## Required Python Libraries

To run the codes provided in this repository, please ensure you have the following Python libraries installed:

- `numpy`
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
