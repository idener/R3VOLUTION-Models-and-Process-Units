# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:37:43 2025

@author: Idener
"""
import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.constants import R, calorie

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:37:43 2025

@author: Idener
"""
import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.constants import R, calorie

class MembraneProcessModel:

    @staticmethod
    def water_properties(T: float) -> Dict[str, float]:
        """
        Calculate temperature-dependent water properties
        Returns: density (kg/m3), viscosity (Pa·s), vapor pressure (Pa), heat capacity (J/kg·K)
        """
        # Density (kg/m3) - Eq from IAPWS
        rho = (999.83952 + 16.945176*T - 7.9870401e-3*T**2 - 46.170461e-6*T**3 
               + 105.56302e-9*T**4 - 280.54253e-12*T**5) / (1 + 16.897850e-3*T)
        
        # Dynamic viscosity (Pa·s) - Vogel equation
        mu = 0.02939 * (10**(247.8/(T + 133.15))) * 1e-3
        
        # Vapor pressure (Pa) - Antoine equation
        Pv = 10**(8.07131 - 1730.63/(233.426 + T)) * 133.322
        
        # Heat capacity (J/kg·K) - Polynomial fit
        Cp = 4217.4 - 3.720283*T + 0.1412855*T**2 - 2.654387e-3*T**3 + 2.093236e-5*T**4
        
        return {'density': rho, 'viscosity': mu, 'vapor_pressure': Pv, 'heat_capacity': Cp}
    
    solute_properties = {
            'fibers_inorganics': {'Mw':50000, 'D': 1.0e-13, 'charge': 0, 'Stokes_radius': 55e-9,'B': 5e-7},
            'pigments': {'Mw': 600, 'D': 5e-10, 'charge': 0, 'Stokes_radius': 5e-6,'B': 5e-9},
            'organics': {'Mw': 10000, 'D': 0.01e-9, 'charge': -1, 'Stokes_radius': 2.0e-9,'B': 1e-9},
            'brine': {'Mw': 23, 'D': 1.33e-9, 'charge': +1, 'Stokes_radius': 0.18e-9, 'B': 2e-8},
            }
    @staticmethod
    def osmotic_pressure(c: float, T: float, phi: float = 1) -> float:
        """
        Calculate osmotic pressure using van't Hoff equation
        c: concentration in mol/m3
        T: temperature in K
        phi: osmotic coefficient (1 for ideal solutions)
        """
        return phi * c * R * (T + 273.15)
    
    @staticmethod
    def mass_transfer_coefficient(flow_rate: float, diameter: float, length: float, 
                                 rho: float, mu: float, D: float) -> float:
        """
        Calculate mass transfer coefficient using Gnielinski correlation
        for turbulent flow in circular channels
        """
        Re = (4 * flow_rate * rho) / (np.pi * diameter * mu)  # Reynolds number
        Sc = mu / (rho * D)  # Schmidt number
        
        if Re < 2300:
            # Laminar flow - Leveque solution
            Sh = 1.62 * (Re * Sc * diameter / length)**(1/3)
        else:
            # Turbulent flow - Gnielinski correlation
            f = (0.79 * np.log(Re) - 1.64)**-2  # friction factor
            Sh = ((f/8)*(Re-1000)*Sc) / (1 + 12.7*np.sqrt(f/8)*(Sc**(2/3)-1)) 
        
        return (Sh * D / diameter)
    
    # CORREGIDO: Función conc_mass_to_mol
    @staticmethod
    def conc_mass_to_mol(feed_conc_dict: Dict[str, float], solute_properties) -> Dict[str, float]:
        """
        Convierte un diccionario de concentraciones en mg/L a mol/m³
        """
        feed_conc_mol_dict = {}
        for comp, conc_mg_L in feed_conc_dict.items():
            comp_key = comp.replace(' ', '_')
            if comp_key in solute_properties:
                Mw = solute_properties[comp_key].get('Mw', 100)
                # mg/L es numéricamente igual a g/m³
                # (g/m³) / (g/mol) = mol/m³
                feed_conc_mol_dict[comp_key] = conc_mg_L / Mw
            else:
                feed_conc_mol_dict[comp_key] = 0.0
                
        return feed_conc_mol_dict
    
    def __init__(self, feed_flow_rate: float, feed_composition: Dict[str, float]):
        """
        Initialize the water reuse system with feed characteristics.
        
        Args:
            feed_flow_rate: Feed flow rate in m3/s
            feed_composition: Dictionary with component concentrations in mg/L
                Expected keys: '2-MeTHF', 'Xylose', 'Lignin', 'Na','Glycoxyllic acid'
        """
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_composition
        
        # Initialize all treatment units
        self.uf = UFUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
    
    def solve_system(self):
        """Solve material and energy balances for the entire system with detailed physics"""
        # Solve each unit in sequence
        self.uf.feed_flow_rate = self.feed_flow_rate
        self.uf.feed_concentration = self.feed_concentration
        self.uf.solve()
        
class UFUnit:
    """Ultrafiltration unit simplified without dynamic effects"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float]):
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        
        # Membrane properties
        self.membrane = 'UF-500'
        self.membrane_thickness = 100e-6
        self.pore_radius = 5e-9
        self.porosity = 0.5
        self.tortuosity = 2.0
        self.A = 5e-11  # Water permeability (m/s·Pa)
        
        # Operating conditions
        self.recovery_ratio = 0.9
        self.transmembrane_pressure = 5.0
        self.temperature = 20
        self.cross_flow_velocity = 0.75
        self.channel_height = 1.5e-3
        
        # Initialize
        self.permeate_flow_rate = None
        self.concentrate_flow_rate = None
        self.permeate_concentration = {}
        self.concentrate_concentration = {}
        self.energy_consumption = None
        self.required_area = None
        self.water_flux = None
       
        self.solute_properties = MembraneProcessModel.solute_properties

    def calculate_rejection(self):
        """Simplified UF rejection model"""
        rejection = {}
        
        for comp, props in self.solute_properties.items():
            # Size-based rejection only
            lambda_ = props['Stokes_radius'] / self.pore_radius
            rejection[comp] = 1 - np.exp(-0.5*lambda_**2)  # Gaussian exclusion model
                
        return rejection
    
    def calculate_flux(self):
        """Simple flux model without osmotic pressure effects"""
        self.water_flux = self.A * self.transmembrane_pressure * 1e5
        return self.water_flux
    
    def solve(self):
        """Simplified solution"""
        
        # Flux calculation
        self.calculate_flux()
        
        # Rejection coefficients
        self.rejection_coeff = self.calculate_rejection()
        
        # Material balance
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Concentration calculations
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            feed_conc = self.feed_concentration[comp]
            
            self.permeate_concentration[comp_key] = feed_conc * (1 - self.rejection_coeff.get(comp_key, 0))
            
            self.concentrate_concentration[comp_key] = (
                (feed_conc * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
        
        # Required membrane area (m²)
        self.required_area = self.permeate_flow_rate / self.water_flux if self.water_flux > 0 else 0
        
        # Energy consumption (kW)
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / 
            (pump_efficiency * 1000)
        )
        
# Example usage
if __name__ == "__main__":
    # Define feed characteristics (typical pulp & paper mill wastewater)
    feed_flow_rate = 1/3600  # m3/s (360 m3/h)
    feed_concentration = {
        'fibers_inorganics': 5000,  # mg/L
        'pigments': 1000,  # mg/L
        'organics': 500,  # mg/L (regulators, brighteners, biocides)
        'brine': 2000  # mg/L (dissolved salts)
    }
    
    
    system = MembraneProcessModel(feed_flow_rate=feed_flow_rate, feed_composition=feed_concentration)
    system.solve_system()

