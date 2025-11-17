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
- Temperature-dependent property calculations

### Felix_PM.py
Paper industry wastewater treatment model featuring:
1. **Microfiltration (MF)**: Concentrates pulp fibers & inorganic compounds
2. **Ultrafiltration (UF)**: Concentrates pigments
3. **Nanofiltration (NF)**: Concentrates organic materials (regulators, brighteners, biocides)
4. **Membrane Distillation (MD)**: Produces clean water and concentrated brine

## Process Units

Individual process unit models that can be integrated into custom treatment trains:

- **RO.py**: Reverse Osmosis
- **NF.py**: Nanofiltration
- **UF.py**: Ultrafiltration
- **MF.py**: Microfiltration
- **MD.py**: Membrane Distillation
- **BPED_4.py**: Bipolar Electrodialysis
- **Chemical Precipitation.py**
- **Neutralization.py**

## Dependencies

```python
numpy
scipy
matplotlib
```

## Usage

Each model can be imported and used independently:

```python
from Models.Felix_PM import MembraneProcessModel

# Define feed characteristics (typical pulp & paper mill wastewater)
feed_flow_rate = 40/3600  # m3/s (360 m3/h)
feed_concentration = {
    'fibers_inorganics': 5000,  # mg/L
    'pigments': 1000,  # mg/L
    'organics': 500,  # mg/L (regulators, brighteners, biocides)
    'brine': 2000  # mg/L (dissolved salts)
}


system = MembraneProcessModel(feed_flow_rate, feed_composition)
system.solve_system()
performance=system.get_system_performance()
```

## Author

@author: Juan Francisco Gutierrez García (Idener)