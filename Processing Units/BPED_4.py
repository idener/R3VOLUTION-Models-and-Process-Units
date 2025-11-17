# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:27:35 2025

@author: Idener
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:27:35 2025

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
        self.bped = BPEDUnit(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        self.feed_pH = feed_pH
        
        
        
    def solve_system(self):
        self.bped.feed_flow_rate = self.feed_flow_rate
        self.bped.feed_concentration = self.feed_concentration
        self.bped.feed_pH = self.feed_pH
        # Pasar las propiedades de los solutos al BPEDUnit
        self.bped.solute_properties = self.solute_properties
        self.bped.solve()
        
class BPEDUnit:
    """Bipolar Electrodialysis unit with  mass balance"""
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate      # m3/s
        self.feed_concentration = feed_concentration  # mg/L (g/m3)
        self.feed_pH = feed_pH
        
        # EDBM-specific operational parameters
        self.cell_pair_number = 50       # Number of cell pairs in the stack
        self.voltage_per_cell_pair = 2.5 # Voltage per cell pair (V)
        self.current_efficiency = 0.85   # Faradaic efficiency for ion transport
        self.bpm_efficiency = 0.90       # Current efficiency for water splitting in BPM
        self.target_ion_removal = 0.95   # Target removal fraction for ions
        self.recovery_ratio = 0.90       # Water recovery (fraction of feed that becomes diluate)
        self.pumping_energy_per_m3 = 50000 # Pumping energy (J/m3 of feed)
        self.temperature = 25            # Operating temperature (°C)
        self.current_density = 100.0   # A/m^2 (Typical range 50-200)
        
        # Initialize results
        self.diluate_flow_rate = None      # (Deionized) water product flow rate (m3/s)
        self.concentrate_flow_rate = None  # Concentrate (brine) flow rate (m3/s)
        self.diluate_concentration = {}    # Diluate (product) concentration (mg/L)
        self.concentrate_concentration = {} # Concentrate (brine) concentration (mg/L)
        self.chemical_production = {}      # Acid/Base production (kg/s)
        self.energy_consumption = None     # Total energy consumption (kW)
        self.solute_properties = SteelWastewaterReuseSystem.solute_properties
        self.total_current = 0             # Total stack current (A)
        self.membrane_area = None          # Required active area per membrane (m^2)
        
    def calculate_ion_transport(self) -> float:
        """
        Calculates the ionic mass balance based on the target removal.
        This defines the diluate and concentrate streams.
        
        Returns:
            float: Total rate of equivalents removed (eq/s)
        """
        total_eq_removed_per_s = 0
        
        for metal, conc_g_m3 in self.feed_concentration.items():
            metal_key = metal.replace(' ', '_')
            
            if metal_key in self.solute_properties:
                # Get properties
                Mw = self.solute_properties[metal_key]['Mw']     # g/mol
                charge = abs(self.solute_properties[metal_key]['charge'])
                
                # (g/m3) * (m3/s) = g/s (inlet mass flow)
                mass_flow_in_g_s = conc_g_m3 * self.feed_flow_rate
                
                # (g/s) / (g/mol) = mol/s (inlet molar flow)
                molar_flow_in_mol_s = mass_flow_in_g_s / Mw
                
                # mol/s removed
                molar_flow_removed_mol_s = molar_flow_in_mol_s * self.target_ion_removal
                
                # (mol/s) * (eq/mol) = eq/s removed
                total_eq_removed_per_s += molar_flow_removed_mol_s * charge
                
                # ---- Calculate outlet concentrations ----
                
                # Outlet mass in diluate (mg/L or g/m3)
                self.diluate_concentration[metal] = conc_g_m3 * (1 - self.target_ion_removal)
                
                # Mass removed (g/s) - this goes to the concentrate
                mass_removed_g_s = mass_flow_in_g_s * self.target_ion_removal
                
                # Concentration in concentrate (mg/L or g/m3)
                if self.concentrate_flow_rate > 0:
                    # (g/s) / (m3/s) = g/m3
                    self.concentrate_concentration[metal] = mass_removed_g_s / self.concentrate_flow_rate
                else:
                    self.concentrate_concentration[metal] = 0
            
            else:
                # If solute is not in the list, assume it is not transported
                self.diluate_concentration[metal] = conc_g_m3
                self.concentrate_concentration[metal] = 0
                
        return total_eq_removed_per_s
    
    def solve(self):
        """Solves the mass and energy balance for the EDBM unit"""
            
        FARADAY_CONSTANT = 96485
        
        # 1. Flow Rates
        self.diluate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate * (1 - self.recovery_ratio)
        
        # 2. Mass Balance and Equivalents Calculation
        total_eq_removed_s = self.calculate_ion_transport()
        
        # 3. Current Calculation
        # I = (Flux_eq * F) / (eta_ion * N_pairs)
        # This is where current_efficiency and cell_pair_number are used
        if self.current_efficiency > 0 and self.cell_pair_number > 0:
            self.total_current = (total_eq_removed_s * FARADAY_CONSTANT) / \
                                 (self.current_efficiency * self.cell_pair_number)
        else:
            self.total_current = 0
        
        # Calculate required membrane area based on current density
        # Area = Total_Current / Current_Density
        if self.current_density > 0:
            self.membrane_area = self.total_current / self.current_density
        else:
            self.membrane_area = 0 # Avoid division by zero
            
        # 4. Chemical Production (Acid/Base via BPM)
        moles_H_produced_s = (self.total_current * self.cell_pair_number * self.bpm_efficiency) / FARADAY_CONSTANT
        moles_OH_produced_s = (self.total_current * self.cell_pair_number * self.bpm_efficiency) / FARADAY_CONSTANT
        
        kg_HCl_s = moles_H_produced_s * 0.03646 
        kg_NaOH_s = moles_OH_produced_s * 0.040
        
        self.chemical_production = {
            'HCl_produced': kg_HCl_s,
            'NaOH_produced': kg_NaOH_s
        }
        
        # 5. Energy Consumption (kW)
        # P = V * I = (V_per_cell * N_cells) * I_total
        # This is where voltage_per_cell_pair is used
        power_stack_W = (self.voltage_per_cell_pair * self.cell_pair_number) * self.total_current
        power_stack_kW = power_stack_W / 1000
        
        power_pump_W = self.pumping_energy_per_m3 * self.feed_flow_rate
        power_pump_kW = power_pump_W / 1000
        
        self.energy_consumption = power_stack_kW + power_pump_kW

    
        
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
