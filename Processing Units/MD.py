# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:04:48 2025

@author: Idener
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:04:48 2025

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
            '2-MeTHF': {'Mw': 86.13, 'D': 1.0e-9, 'charge': 0, 'Stokes_radius': 0.3e-9,'B': 1e-8},
            'Xylose': {'Mw': 150.13, 'D': 0.67e-9, 'charge': 0, 'Stokes_radius': 0.45e-9,'B': 5e-9},
            'Lignin': {'Mw': 10000, 'D': 0.01e-9, 'charge': -1, 'Stokes_radius': 2.0e-9,'B': 1e-9},
            'Na': {'Mw': 23, 'D': 1.33e-9, 'charge': +1, 'Stokes_radius': 0.18e-9, 'B': 2e-8},
            'Glycoxyllic_acid': {'Mw': 74.04, 'D': 0.9e-9, 'charge': -1, 'Stokes_radius': 0.35e-9, 'B': 3e-9},
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
        self.md = MDUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
    
    def solve_system(self):
        """Solve material and energy balances for the entire system with detailed physics"""
        # Solve each unit in sequence
        self.md.feed_flow_rate = self.feed_flow_rate
        self.md.feed_concentration = self.feed_concentration
        self.md.solve()
        
class MDUnit:
    """Membrane Distillation unit with Dusty Gas Model"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float]):
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        
        # Membrane properties
        self.membrane = 'PE1'
        self.membrane_thickness = 68e-6  # m
        self.pore_size = 0.1e-6  # m
        self.porosity = 0.6
        self.tortuosity = 1.5
        
        # Operating conditions
        self.recovery_ratio = 0.72
        self.feed_temp = 80 # °C
        self.permeate_temp = 50  # °C
        self.cross_flow_velocity = 0.2  # m/s
        
        # Initialize calculated properties
        self.permeate_flow_rate = None
        self.concentrate_flow_rate = None
        self.permeate_concentration = {}
        self.concentrate_concentration = {}
        self.thermal_energy = None
        self.electrical_energy = None
        self.required_area = None
        self.J = None
        
    def solve(self):
        """Solve MD system using Dusty Gas Model"""
        feed_props = MembraneProcessModel.water_properties(self.feed_temp)
        permeate_props = MembraneProcessModel.water_properties(self.permeate_temp)
        
        # Vapor pressures (Pa)
        P_f = feed_props['vapor_pressure']
        P_p = permeate_props['vapor_pressure']
    
        self.Diff_Knudsen = (1/3) * self.pore_size* np.sqrt((8*R*(self.feed_temp+273)/(np.pi*18e3)))
        C= self.porosity/(self.tortuosity*self.membrane_thickness)*self.Diff_Knudsen
        self.J = C * (P_f - P_p)/feed_props['density'] #m3/m2s
        
        # Permeate flow rate (m3/s)
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio

        # Material balance
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Concentration calculations (assuming complete rejection of non-volatiles)
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            feed_conc = self.feed_concentration[comp] # Conc. de entrada
            
            self.permeate_concentration[comp_key] = 0.01 * feed_conc  # Ideal MD rejects all non-volatiles
            
            # CORREGIDO: Balance de masas robusto
            self.concentrate_concentration[comp_key] = (
                (feed_conc * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
        
        # Energy calculations
        latent_heat = 2.26e6  # J/kg
        Cp = feed_props['heat_capacity']  # J/kg·K
        
        # Thermal energy (W)
        self.thermal_energy = (
            self.permeate_flow_rate * latent_heat + 
            self.feed_flow_rate * feed_props['density'] * Cp * (self.feed_temp - 25)
        )
        # Electrical energy (pumping, kW)
        pump_efficiency = 0.7
        self.electrical_energy = (
            self.feed_flow_rate * 2 * 1e5 / 
            (pump_efficiency * 1000)
        )
        
        # Required membrane area (m2)
        if not self.required_area and self.J > 0:
            self.required_area = (self.permeate_flow_rate / self.J)
        elif not self.required_area:
            self.required_area = 0
        
if __name__ == "__main__":
    # Define feed characteristics 
    feed_flow_rate =1/3600  # m3/s (1 m3/h)
    feed_concentration_input = {
        '2-MeTHF': 2.68,  # mg/L
        'Xylose': 1000,  # mg/L
        'Lignin': 0.15,  # mg/L 
        'Na': 12900,  # mg/L 
        'Glycoxyllic_acid': 4.08  # mg/L 
    }
    
    feed_concentration= { '2-MeTHF': feed_concentration_input['2-MeTHF']/100*0.867*1000*1000,  # mg/L
                         'Xylose': feed_concentration_input['Xylose'],  # mg/L
                         'Lignin': feed_concentration_input['Lignin']/100*1000*1000,  # mg/L 
                         'Na': feed_concentration_input['Na']/1000,  # mg/L 
                         'Glycoxyllic_acid': feed_concentration_input['Glycoxyllic_acid']/100*1000*1000  # mg/L 
    }
    
        # Create and solve the water reuse system
    system = MembraneProcessModel(feed_flow_rate=feed_flow_rate, feed_composition=feed_concentration)
    system.solve_system()

