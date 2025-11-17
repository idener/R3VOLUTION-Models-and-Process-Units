# R3VOLUTION Models and Process Units

A comprehensive Python library for modeling and simulating water treatment and wastewater reuse systems, with a focus on membrane-based separation processes and industrial applications.

## Overview

This repository contains advanced physical models and simulation tools for various water treatment processes, including membrane filtration, chemical precipitation, and electrodialysis. The models incorporate detailed thermodynamic equations, heat and mass transfer correlations, and cost estimation capabilities for industrial wastewater treatment and water reuse applications.

## Repository Structure

```
R3VOLUTION-Models-and-Processing-Units/
├── Models/                      # Complete system models for specific applications
│   ├── BLOOM_PM.py             # Enhanced membrane process model (NF, RO, MD)
│   ├── CELSA_PM_4.py           # Steel wastewater reuse system model
│   └── Felix_PM.py             # Paper industry wastewater treatment model
└── Process Units/               # Individual process unit models
    ├── BPED_4.py               # Bipolar Electrodialysis (BPED)
    ├── Chemical Precipitation.py
    ├── MD.py                   # Membrane Distillation
    ├── MF.py                   # Microfiltration
    ├── NF.py                   # Nanofiltration
    ├── Neutralization.py
    ├── RO.py                   # Reverse Osmosis
    └── UF.py                   # Ultrafiltration
```

## Models

### BLOOM_PM.py
Enhanced membrane process model with detailed physical equations for:
- **Nanofiltration (NF)**: Solution-diffusion model with Extended Nernst-Planck for charged species
- **Reverse Osmosis (RO)**: High-pressure water purification
- **Membrane Distillation (MD)**: Dusty gas model for thermal separation
- Temperature-dependent water properties (density, viscosity, vapor pressure)
- Heat and mass transfer correlations
- Detailed cost models

### CELSA_PM_4.py
Comprehensive steel wastewater reuse system model including:
- Multiple membrane processes for steel industry applications
- Physical and chemical treatment units
- Cost optimization and economic analysis
- Temperature-dependent property calculations

### Felix_PM.py
Paper industry wastewater treatment model featuring:
1. **Microfiltration (MF)**: Concentrates pulp fibers & inorganic compounds
2. **Ultrafiltration (UF)**: Concentrates pigments
3. **Nanofiltration (NF)**: Concentrates organic materials (regulators, brighteners, biocides)
4. **Membrane Distillation (MD)**: Produces clean water and concentrated brine

## Process Units

Individual process unit models that can be integrated into custom treatment trains:

- **RO.py**: Reverse Osmosis - High-pressure membrane filtration for water purification
- **NF.py**: Nanofiltration - Selective separation of multivalent ions and small organic molecules
- **UF.py**: Ultrafiltration - Removal of macromolecules, colloids, and suspended solids
- **MF.py**: Microfiltration - Particle and bacteria removal
- **MD.py**: Membrane Distillation - Thermal desalination process
- **BPED_4.py**: Bipolar Electrodialysis - Acid and base production from salt solutions
- **Chemical Precipitation.py**: Chemical treatment for contaminant removal
- **Neutralization.py**: pH adjustment and neutralization processes

## Dependencies

```python
numpy
scipy
matplotlib
```

## Usage

Each model can be imported and used independently:

```python
from Models.BLOOM_PM import MembraneProcessModel
from Process_Units.RO import MembraneProcessModel as RO

# Initialize model
model = MembraneProcessModel()

# Calculate water properties at 25°C
props = model.water_properties(25)
print(f"Density: {props['density']} kg/m³")
print(f"Viscosity: {props['viscosity']} Pa·s")
```

## Author

@author: Juan Francisco Gutierrez García (Idener)