# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:55:35 2025

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
        self.ro = ROUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
    
    def solve_system(self):
        """Solve material and energy balances for the entire system with detailed physics"""
        # Solve each unit in sequence
        self.ro.feed_flow_rate = self.feed_flow_rate
        self.ro.feed_concentration = self.feed_concentration
        self.ro.solve()
        
class ROUnit:
    """Reverse Osmosis unit with solution-diffusion model and concentration polarization"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float]):
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        self.solute_properties=MembraneProcessModel.solute_properties
        # Membrane properties
        self.membrane = 'AFC99'
        self.membrane_thickness = 0.1e-6  # m (typical RO membrane)
        self.A_water = 2e-11  # Water permeability (m/s·Pa)
     
        # Operating conditions
        self.recovery_ratio = 0.84
        self.transmembrane_pressure = 30  # bar
        self.temperature = 25  # °C
        self.cross_flow_velocity = 0.5  # m/s
        self.channel_height = 0.7e-3  # m (typical RO spacer height)
     
        # Initialize calculated properties
        self.permeate_flow_rate = None
        self.concentrate_flow_rate = None
        self.permeate_concentration = {}
        self.concentrate_concentration = {}
        self.energy_consumption = None
        self.required_area = None
        self.Jw = None

    # CORREGIDO: Lógica de 'solve' reemplazada por el solver iterativo robusto
    def solve(self):
        """Solve RO system using solution-diffusion model with CP"""
        water_props = MembraneProcessModel.water_properties(self.temperature)
        
        # Flujos
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Diccionario de concentraciones de alimentación en mol/m³
        conc_mol_dict = MembraneProcessModel.conc_mass_to_mol(
            self.feed_concentration, 
            self.solute_properties
        )
        
        # Calcular coeficientes de transferencia de masa (k) para todos los solutos
        k_dict = {}
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            D_value = self.solute_properties[comp_key].get('D', 1e-9)
            k_dict[comp_key] = MembraneProcessModel.mass_transfer_coefficient(
                self.feed_flow_rate,
                self.channel_height,
                1.0,  # length placeholder
                water_props['density'],
                water_props['viscosity'],
                D_value
            )

        #--- Solución iterativa para Jw (flujo de agua) y C_m (concentración en membrana) ---
        tolerance = 1e-6
        max_iter = 100
        
        # Estimación inicial de Jw (sin presión osmótica)
        self.Jw = self.A_water * (self.transmembrane_pressure * 1e5) 
        
        C_m_dict = conc_mol_dict.copy()
        C_p_dict = {key: 0.0 for key in conc_mol_dict}

        for _ in range(max_iter):
            # 1. Calcular C_m (superficie de membrana) para todos los solutos basado en Jw actual
            for comp_key in C_m_dict:
                feed_conc_mol = conc_mol_dict.get(comp_key, 0)
                k = k_dict.get(comp_key, 1e-5)
                C_m_dict[comp_key] = feed_conc_mol * np.exp(self.Jw / k) if k > 0 else feed_conc_mol
            
            # 2. Calcular C_p (permeado) para todos los solutos
            for comp_key in C_p_dict:
                B = self.solute_properties[comp_key].get('B', 1e-8) # Permeabilidad del soluto
                Js = B * (C_m_dict[comp_key] - C_p_dict.get(comp_key, 0)) # Flujo de soluto
                C_p_dict[comp_key] = Js / self.Jw if self.Jw > 0 else 0
            
            # 3. Calcular presión osmótica total en ambos lados
            pi_m = MembraneProcessModel.osmotic_pressure(sum(C_m_dict.values()), self.temperature)
            pi_p = MembraneProcessModel.osmotic_pressure(sum(C_p_dict.values()), self.temperature)
            delta_pi = pi_m - pi_p
            
            # 4. Calcular nuevo Jw
            new_Jw = self.A_water * (self.transmembrane_pressure * 1e5 - delta_pi)
            
            # 5. Comprobar convergencia
            if abs(new_Jw - self.Jw) / (self.Jw + 1e-9) < tolerance:
                self.Jw = new_Jw
                break
            
            self.Jw = new_Jw
        else:
            print(f"Advertencia: ROUnit.solve no convergió después de {max_iter} iteraciones.")
        
        #--- Fin de la iteración ---

        # Guardar concentraciones finales
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            Mw = MembraneProcessModel.solute_properties[comp_key]['Mw']
            
            # CORREGIDO: Conversión de C_p (mol/m³) a mg/L (eliminado * 1e3)
            # (mol/m³) * (g/mol) = g/m³ (que es == mg/L)
            self.permeate_concentration[comp_key] = C_p_dict.get(comp_key, 0) * Mw
            
            # Calcular concentración del concentrado
            feed_conc_mgL = self.feed_concentration[comp]
            
            # CORREGIDO: Eliminado el 'max(0, ...)'
            self.concentrate_concentration[comp_key] = (
                (feed_conc_mgL * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
            
        # Calcular área de membrana requerida
        if self.Jw and self.Jw > 0:
            self.required_area = self.permeate_flow_rate / self.Jw
        else:
            self.required_area = 0
        
        # Consumo de energía
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / (pump_efficiency * 1000)
        )        
        
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

