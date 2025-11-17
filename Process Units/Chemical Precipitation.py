# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:56:31 2025

@author: Idener
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.constants import R, calorie


class SteelWastewaterReuseSystem:
    
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
            'Fe': {'Mw': 55.85, 'D': 6.0e-10, 'charge': 3, 'Stokes_radius': 4.5e-10,'B': 1e-8},
            'Mn': {'Mw': 54.94, 'D': 7e-10, 'charge': 2, 'Stokes_radius': 4e-10,'B': 1e-8},
            'Zn': {'Mw': 65.38, 'D': 7.1e-10, 'charge': 2, 'Stokes_radius': 4e-10,'B': 1e-8},
            'Ni': {'Mw': 58.69, 'D': 6.9e-10, 'charge': 2, 'Stokes_radius': 4.1e-10, 'B': 1e-8},
            'Cr': {'Mw': 51.996, 'D': 5.8e-10, 'charge': 3, 'Stokes_radius': 4.5e-10, 'B': 1e-8},
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
    
    @staticmethod
    def conc_mass_to_mol(feed_conc: Dict[str, float], solute_properties) -> Dict[str, float]:
        """
        Convert mass concentration (mg/L) to molar concentration (mol/m3)
        """
        feed_conc_mol = {}
        
        for comp, conc_value in feed_conc.items():
            comp_key = comp.replace(' ', '_')
            
            # Convert to mol/m3
            if comp_key in solute_properties and 'Mw' in solute_properties[comp_key]:
                Mw = solute_properties[comp_key]['Mw']
            else:
                Mw = 100  # default molecular weight
                
            # mg/L to mol/m3 conversion: (mg/L) * (1g/1000mg) * (1mol/Mw g) * (1000L/1m3) = mol/m3
            feed_conc_mol[comp_key] = conc_value / Mw
            
        return feed_conc_mol
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH):
        """
        Initialize the water reuse system with feed characteristics.
        
        Args:
            feed_flow_rate: Feed flow rate in m3/s
            feed_composition: Dictionary with component concentrations in mg/L
        """
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        self.feed_pH = feed_pH
        
        self.cp = ChemicalPrecipitation(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        
        
    def solve_system(self):
        self.cp.feed_flow_rate = self.feed_flow_rate
        self.cp.feed_concentration = self.feed_concentration
        self.cp.feed_pH = self.feed_pH
        self.cp.solve()
        
        
class ChemicalPrecipitation:
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate  # m3/s
        self.feed_concentration = feed_concentration  # mg/L
        self.feed_pH = feed_pH
        
        self.lime_dose = 0.5  # kg/m3
        self.coagulant_dose = 0.2  # kg/m3
        self.dosing_ratio = 1.5
        self.precipitant = 'NaOH'  # Primary precipitant
        self.coagulant = 'FeCl3'  # Coagulant aid
        self.flocculant_dose = 0.05  # kg/m3
        self.water_recovery = 0.9
        self.metal_removal = 0.95
        self.pH_target = 9
        # Operating conditions
        self.mixing_energy = 0.1  # kW/m³
        self.settling_time = 60  # min
        self.temperature = 25  # °C
        self.recovery_ratio = 0.95
        
        # Initialize results
        self.treated_flow_rate = None
        self.sludge_flow_rate = None
        self.treated_concentration = {}
        self.sludge_concentration = {}
        self.chemical_consumption = None
        self.energy_consumption = None
        self.solute_properties = SteelWastewaterReuseSystem.solute_properties
        
    def calculate_precipitation_efficiency(self):
        """Calculate precipitation efficiency based on solubility products and pH"""
        removal_efficiency = {}
        
        # Solubility characteristics for steel industry metals
        precipitation_pH = {
            'Fe': {'optimal_pH': 8.5, 'min_pH': 6.5, 'max_efficiency': 0.995},
            'Mn': {'optimal_pH': 10.0, 'min_pH': 9.0, 'max_efficiency': 0.98},
            'Zn': {'optimal_pH': 9.0, 'min_pH': 7.5, 'max_efficiency': 0.99},
            'Ni': {'optimal_pH': 10.5, 'min_pH': 9.5, 'max_efficiency': 0.97},
            'Cr': {'optimal_pH': 8.0, 'min_pH': 7.0, 'max_efficiency': 0.99}
        }
        
        for metal, props in precipitation_pH.items():
            if metal in self.feed_concentration:
                if self.pH_target >= props['optimal_pH']:
                    # Optimal conditions
                    removal_efficiency[metal] = props['max_efficiency']
                elif self.pH_target >= props['min_pH']:
                    # Sub-optimal but effective
                    efficiency_range = props['max_efficiency'] - 0.70
                    pH_range = props['optimal_pH'] - props['min_pH']
                    removal_efficiency[metal] = 0.70 + (efficiency_range * 
                                                       (self.pH_target - props['min_pH']) / pH_range)
                else:
                    # Below minimum effective pH
                    removal_efficiency[metal] = 0.30  # Limited precipitation
        
        return removal_efficiency
    
    def calculate_chemical_consumption(self):
        """Calculate chemical consumption based on stoichiometry"""
        total_metal_equivalents = 0  # mol/s
        
        for metal, conc in self.feed_concentration.items():
            metal_key = metal.replace(' ', '_')
            if metal_key in self.solute_properties:
                # Convert mg/L to mol/s
                molar_mass = self.solute_properties[metal_key]['Mw']
                charge = abs(self.solute_properties[metal_key]['charge'])
                mol_per_second = (conc * self.feed_flow_rate) / molar_mass  # mol/s
                total_metal_equivalents += mol_per_second * charge
        
        # NaOH consumption (kg/s)
        naoh_consumption = total_metal_equivalents * 0.040 * self.dosing_ratio
        
        # Coagulant consumption (typically 10-50 mg/L)
        coagulant_dose = 30  # mg/L
        coagulant_consumption = coagulant_dose * self.feed_flow_rate * 1e-3  # kg/s
        
        return {'NaOH': naoh_consumption, 'FeCl3': coagulant_consumption}
    
    def solve(self):
        """Solve precipitation mass and energy balance"""
        # Flow rates
        self.treated_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.sludge_flow_rate = self.feed_flow_rate * (1 - self.recovery_ratio)
        
        # Removal efficiencies
        removal_efficiencies = self.calculate_precipitation_efficiency()
        
        # Concentration calculations
        sludge_solids = 0
        for metal, conc in self.feed_concentration.items():
            if metal in removal_efficiencies:
                removal = removal_efficiencies[metal]
                self.treated_concentration[metal] = conc * (1 - removal)
                
                # Sludge concentration (mg/L in sludge stream)
                metal_in_sludge = conc * removal * self.feed_flow_rate
                if self.sludge_flow_rate > 0:
                    self.sludge_concentration[metal] = metal_in_sludge / self.sludge_flow_rate
                else:
                    self.sludge_concentration[metal] = 0
                sludge_solids += self.sludge_concentration[metal]
            else:
                self.treated_concentration[metal] = conc
                self.sludge_concentration[metal] = 0
        
        # Chemical consumption (kg/s)
        chemical_consumption = self.calculate_chemical_consumption()
        self.chemical_consumption = chemical_consumption
        
        # Energy consumption (kW)
        self.energy_consumption = self.mixing_energy * self.feed_flow_rate * 1000  # kW
        
if __name__ == "__main__":
    # Define feed water characteristics
    feed_flow = 68.2/3600  # m3/s (1 m3/h)
    feed_conc = {
        'Fe': 500,  # mg/L
        'Mn': 100,
        'Zn': 50,
        'Ni': 20,
        'Cr': 10,
    }
    feed_pH = 7
    
    # Create and solve system
    system = SteelWastewaterReuseSystem(feed_flow, feed_conc, feed_pH)
    system.solve_system()