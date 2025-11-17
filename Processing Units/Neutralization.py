# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:03:48 2025

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
        
        self.nt = Neutralization(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        
        
    def solve_system(self):
        self.nt.feed_flow_rate = self.feed_flow_rate
        self.nt.feed_concentration = self.feed_concentration
        self.nt.feed_pH = self.feed_pH
        self.nt.solve()
        
        
class Neutralization:
    """Acid-base neutralization unit for pH adjustment in steel wastewater"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate  # m3/s
        self.feed_concentration = feed_concentration  # mg/L
        self.feed_pH = feed_pH  # pH inicial de la alimentación
        
        # Neutralization chemicals (selección automática basada en pH)
        self.target_pH = 7.0  # pH objetivo neutro
        
        # Determinar agente neutralizante según pH inicial 
        if self.feed_pH is not None and self.feed_pH < 7.0:
            self.neutralizing_agent = 'NaOH'  # Para aguas ácidas
        elif self.feed_pH is not None:
            self.neutralizing_agent = 'H2SO4'  # Para aguas básicas
        else:
            self.neutralizing_agent = 'NaOH'  # Valor por defecto si es None
        
        # Operating conditions
        self.mixing_energy = 0.05  # kW/m³
        self.reaction_time = 15  # min
        self.temperature = 25  # °C
        
        # Initialize results
        self.treated_flow_rate = None
        self.treated_concentration = {}
        self.chemical_consumption = None
        self.energy_consumption = None
        self.treated_pH = None
        
        self.solute_properties = SteelWastewaterReuseSystem.solute_properties
        
    def calculate_acidity_alkalinity(self):
        """Calculate net acidity or alkalinity considering metals and pH"""
        # Acidez de metales (mol H+/m³)
        metal_acidity_factors = {
            'Fe': 3,  # Fe³⁺ produce acidez al hidrolizarse
            'Mn': 2,  
            'Zn': 2,  
            'Ni': 2,  
            'Cr': 3   
        }
        
        metal_acidity = 0
        for metal, conc in self.feed_concentration.items():
            metal_key = metal.replace(' ', '_')
            if metal_key in metal_acidity_factors and metal_key in self.solute_properties:
                molar_mass = self.solute_properties[metal_key]['Mw']
                moles_per_m3 = conc / molar_mass
                metal_acidity += moles_per_m3 * metal_acidity_factors[metal]
        
        # Acidez/alcalinidad neta basada en pH
        H_conc = 10**(-self.feed_pH) * 1000  # mol H+/m³
        OH_conc = 10**(-(14 - self.feed_pH)) * 1000  # mol OH-/m³
        
        if self.feed_pH < 7.0:
            # Agua ácida - net acidity
            net_acidity = metal_acidity + H_conc - OH_conc
            return max(0, net_acidity), 0  # (acidez, alcalinidad)
        else:
            # Agua básica - net alkalinity
            net_alkalinity = OH_conc - H_conc  # Alcalinidad de hidróxidos
            return 0, max(0, net_alkalinity)  # (acidez, alcalinidad)
        
    def calculate_chemical_dose(self):
        """Calculate required chemical dose for neutralization in both directions"""
        acidity, alkalinity = self.calculate_acidity_alkalinity()
        
        # Concentración objetivo
        target_H = 10**(-self.target_pH) * 1000  # mol/m³
        target_OH = 10**(-(14 - self.target_pH)) * 1000  # mol/m³
        
        if self.feed_pH < 7.0:
            # Neutralización de acidez
            H_initial = 10**(-self.feed_pH) * 1000
            required_neutralization = max(0, acidity + (H_initial - target_H))
            
            if self.neutralizing_agent == 'NaOH':
                dose_kg_per_m3 = required_neutralization * 0.040  # 40 g/mol
            elif self.neutralizing_agent == 'Ca(OH)2':
                dose_kg_per_m3 = required_neutralization * 0.074 / 2  # 74 g/mol, 2 eq
            else:
                dose_kg_per_m3 = required_neutralization * 0.040
                
        else:
            # Neutralización de alcalinidad (acidificación)
            OH_initial = 10**(-(14 - self.feed_pH)) * 1000
            required_acidification = max(0, alkalinity + (OH_initial - target_OH))
            
            if self.neutralizing_agent == 'H2SO4':
                dose_kg_per_m3 = required_acidification * 0.098 / 2  # 98 g/mol, 2 eq
            elif self.neutralizing_agent == 'HCl':
                dose_kg_per_m3 = required_acidification * 0.0365  # 36.5 g/mol, 1 eq
            else:
                dose_kg_per_m3 = required_acidification * 0.098 / 2
                
        return dose_kg_per_m3

    def solve(self):
        """Solve neutralization mass and energy balance"""
        self.treated_flow_rate = self.feed_flow_rate
        
        # Chemical consumption (kg/s)
        dose_kg_per_m3 = self.calculate_chemical_dose()
        self.chemical_consumption = {
            self.neutralizing_agent: dose_kg_per_m3 * self.feed_flow_rate
        }
        
        # Energy consumption (kW)
        self.energy_consumption = self.mixing_energy * self.feed_flow_rate * 1000
        
        # Metal concentrations remain unchanged
        for metal, conc in self.feed_concentration.items():
            self.treated_concentration[metal] = conc
        
        self.treated_pH = self.target_pH

    def get_neutralization_info(self):
        """Return information about the neutralization process"""
        acidity, alkalinity = self.calculate_acidity_alkalinity()
        
        self.info = {
            'feed_pH': self.feed_pH,
            'target_pH': self.target_pH,
            'neutralizing_agent': self.neutralizing_agent,
            'process_type': 'Acid neutralization' if self.feed_pH < 7.0 else 'Alkali neutralization',
            'net_acidity': acidity,
            'net_alkalinity': alkalinity
        }
        return self.info
        
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
    feed_pH = 10
    
    # Create and solve system
    system = SteelWastewaterReuseSystem(feed_flow, feed_conc, feed_pH)
    system.solve_system()