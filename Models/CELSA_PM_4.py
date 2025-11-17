# -*- coding: utf-8 -*-
"""

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
            # Esta conversión (conc_value / Mw) es correcta porque mg/L == g/m³
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
        
        #Initialize all treatment units
        self.uf = UFUnit(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        self.nf = NFUnit(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        self.ro = ROUnit(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        self.md = MDUnit(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        self.bped = BPEDUnit(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        self.cp = ChemicalPrecipitation(feed_flow_rate=None, feed_concentration={}, feed_pH=None)
        self.nt = Neutralization(feed_flow_rate=None, feed_concentration={}, feed_pH=None)

    def solve_system(self):
        # UF Unit
        self.uf.feed_flow_rate = self.feed_flow_rate
        self.uf.feed_concentration = self.feed_concentration
        self.uf.feed_pH = self.feed_pH
        self.uf.solve()
        
        # NF Unit
        self.nf.feed_flow_rate = self.uf.permeate_flow_rate
        self.nf.feed_concentration = self.uf.permeate_concentration
        self.nf.feed_pH = self.feed_pH
        self.nf.solve()
        
        # RO Unit
        self.ro.feed_flow_rate = self.nf.permeate_flow_rate
        self.ro.feed_concentration = self.nf.permeate_concentration
        self.ro.feed_pH = self.feed_pH
        self.ro.solve()
        
        # MD Unit
        self.md.feed_flow_rate = self.ro.concentrate_flow_rate
        self.md.feed_concentration = self.ro.concentrate_concentration
        self.md.feed_pH = self.feed_pH
        self.md.solve()
        
        # BPED Unit
        self.bped.feed_flow_rate = self.md.concentrate_flow_rate
        self.bped.feed_concentration = self.md.concentrate_concentration
        self.bped.feed_pH = self.feed_pH
        self.bped.solve()
        
        # Chemical Precipitation
        self.cp.feed_flow_rate = self.uf.concentrate_flow_rate + self.nf.concentrate_flow_rate
        self.cp.feed_concentration = {}
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            uf_conc = self.uf.concentrate_concentration.get(comp_key, 0)
            nf_conc = self.nf.concentrate_concentration.get(comp_key, 0)
            uf_flow = self.uf.concentrate_flow_rate if self.uf.concentrate_flow_rate else 0
            nf_flow = self.nf.concentrate_flow_rate if self.nf.concentrate_flow_rate else 0
            
            if self.cp.feed_flow_rate > 0:
                self.cp.feed_concentration[comp] = (uf_conc * uf_flow + nf_conc * nf_flow) / self.cp.feed_flow_rate
            else:
                self.cp.feed_concentration[comp] = 0
        self.cp.feed_pH = self.feed_pH
        self.cp.solve()
        
        # Neutralization
        self.nt.feed_flow_rate = self.cp.treated_flow_rate
        self.nt.feed_concentration = self.cp.treated_concentration
        self.nt.feed_pH = self.cp.pH_target
        self.nt.solve()
        
        
    def get_system_performance (self) -> Dict:
        """Calculate overall system performance metrics"""
        water_recovery = (self.ro.permeate_flow_rate + self.md.permeate_flow_rate + self.bped.diluate_flow_rate) / self.feed_flow_rate * 100
        
        recycled_water_flow = (self.ro.permeate_flow_rate + self.md.permeate_flow_rate + self.bped.diluate_flow_rate) * 3600
        
        contaminants_removal = {}
        for comp in self.feed_concentration.keys():

            masa_salida = (
                self.ro.permeate_concentration.get(comp, 0) * self.ro.permeate_flow_rate + 
                self.md.permeate_concentration.get(comp, 0) * self.md.permeate_flow_rate + 
                self.bped.diluate_concentration.get(comp, 0) * self.bped.diluate_flow_rate
            )
            masa_entrada = self.feed_concentration[comp] * self.feed_flow_rate
            
            if masa_entrada > 0:
                contaminants_removal[comp] = (1 - (masa_salida / masa_entrada)) * 100
            else:
                contaminants_removal[comp] = 0

        
        electricity_consumption = {'UF (kW)': self.uf.energy_consumption,
                                   'NF (kW)': self.nf.energy_consumption,
                                   'RO (kW)': self.ro.energy_consumption,
                                   'MD (kW)': self.md.electrical_energy                 ,
                                   'BPED (kW)': self.bped.energy_consumption,
                                   'Chemical Precipitation (kW)': self.cp.energy_consumption,
                                   'Neutralization (kW)': self.nt.energy_consumption
                                   }
        membrane_areas_m2 = {'UF': self.uf.required_area,
                             'NF': self.nf.required_area,
                             'RO': self.ro.required_area,
                             'MD': self.md.required_area,
                             'BPED':self.bped.membrane_area}
        
        chemical_production_kg_s = self.bped.chemical_production
        
        chemical_consumption_cp = self.cp.chemical_consumption
        
        neutralizant_agent_consumption = self.nt.chemical_consumption
           
            
        return {'Water recovery (%)': water_recovery,
                'Recycled water flow (m3/h)': recycled_water_flow,
                'Contaminants removal (%)': contaminants_removal,
                'Electrical energy consumption (kW)': electricity_consumption,
                'Membrane units area (m2)': membrane_areas_m2,
                'Chemical production in BPED unit (kg/s)' : chemical_production_kg_s,
                'Chemical consumption in Chemical Precipitation unit (kg/s)':chemical_consumption_cp,
                'Chemical consumption in Neutralization unit (kg/s)': neutralizant_agent_consumption}

            
class UFUnit:
    """Ultrafiltration unit simplified without dynamic effects"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        self.feed_pH = feed_pH
        
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
       
        self.solute_properties = SteelWastewaterReuseSystem.solute_properties

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
        water_props = SteelWastewaterReuseSystem.water_properties(self.temperature)
        
        # Flux calculation
        self.calculate_flux()
        
        # Rejection coefficients
        rejection_coeff = self.calculate_rejection()
        
        # Material balance
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Concentration calculations
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            feed_conc = self.feed_concentration[comp]
            
            self.permeate_concentration[comp_key] = feed_conc * (1 - rejection_coeff.get(comp_key, 0))
            
            if self.concentrate_flow_rate > 0:
                self.concentrate_concentration[comp_key] = (
                    (feed_conc * self.feed_flow_rate - 
                     self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                    self.concentrate_flow_rate
                )
            else:
                self.concentrate_concentration[comp_key] = 0
        
        # Required membrane area (m²)
        if self.water_flux > 0:
            self.required_area = self.permeate_flow_rate / self.water_flux
        else:
            self.required_area = 0
        
        # Energy consumption (kW)
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / 
            (pump_efficiency * 1000)
        )

class NFUnit:
    """Nanofiltration unit with solution-diffusion and Donnan exclusion models"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate  # m3/s
        self.feed_concentration = feed_concentration  # mg/L
        self.feed_pH = feed_pH
        
        # Membrane properties
        self.membrane = 'A-3011'
        self.membrane_thickness = 1e-6  # m (typical NF membrane)
        self.pore_radius = 0.5e-9  # m (for NF)
        self.porosity = 0.2  # -
        self.tortuosity = 2.5  # -
        
        # Operating conditions
        self.recovery_ratio = 0.7
        self.transmembrane_pressure = 20  # bar
        self.temperature = 20  # °C
        self.cross_flow_velocity = 0.5  # m/s
        self.channel_height = 1e-3  # m (typical spacer-filled channel)
        
        # Initialize calculated properties
        self.permeate_flow_rate = None
        self.concentrate_flow_rate = None
        self.permeate_concentration = {}
        self.concentrate_concentration = {}
        self.energy_consumption = None
        self.required_area = None
        self.A = 1e-10  # m/s·Pa Water permeability
        self.water_flux = None
        self.effective_flux = None
        
    def calculate_rejection(self):
        """
        Calculate rejection coefficients using extended Nernst-Planck with Donnan exclusion
        for charged species and steric hindrance for neutral species
        """
        rejection = {}
        self.solute_properties = SteelWastewaterReuseSystem.solute_properties
        
        for comp, props in self.solute_properties.items():
            # Steric hindrance parameter
            lambda_ = props['Stokes_radius'] / self.pore_radius
            
            # CORREGIDO: Añadida la comprobación de exclusión estérica
            if lambda_ >= 1:
                S = 0.0 # Exclusión total
            else:
                S = (1 - lambda_)**2  # steric factor
            
            if props['charge'] != 0:
                # Donnan potential for charged species
                # Simplified Donnan equilibrium (actual implementation would solve Poisson-Boltzmann)
                phi_D = 0.1  # Placeholder for Donnan potential (V)
                K = np.exp(-props['charge'] * phi_D / (R * (self.temperature + 273.15)))
                rejection[comp] = 1 - (S * K)
            else:
                # Neutral species - solution-diffusion model
                # Permeability coefficient (m/s)
                B = (self.porosity * props['D'] * S) / (self.membrane_thickness * self.tortuosity)
                
                # Apparent rejection
                if self.A * self.transmembrane_pressure * 1e5 > 0:
                    rejection[comp] = 1 - (B / (self.A * self.transmembrane_pressure * 1e5))
                else:
                    rejection[comp] = 0
        
        return rejection
    
    def solve(self):
        """Solve material and energy balances with detailed transport models"""
        # Get water properties
        water_props = SteelWastewaterReuseSystem.water_properties(self.temperature)
        
        # Calculate rejections
        rejection_coeff = self.calculate_rejection()
        
        # Material balance
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Concentration calculations
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            feed_conc = self.feed_concentration[comp]  # mg/L
            
            # Permeate concentration (mg/L)
            self.permeate_concentration[comp_key] = (
                feed_conc * (1 - rejection_coeff.get(comp_key, 0))
            ) 
            # Concentrate concentration (mg/L)
            if self.concentrate_flow_rate > 0:
                self.concentrate_concentration[comp_key] = (
                    (feed_conc * self.feed_flow_rate - 
                     self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                    self.concentrate_flow_rate
                )
            else:
                self.concentrate_concentration[comp_key] = 0
        
        # Calcular concentración molar total para presión osmótica
        conc_mol_dict = SteelWastewaterReuseSystem.conc_mass_to_mol(
            self.feed_concentration, 
            SteelWastewaterReuseSystem.solute_properties
        )
        total_conc_mol = sum(conc_mol_dict.values())  # mol/m3
        
        # Mass transfer coefficient (m/s)
        avg_D = 6e-10  # valor promedio
        k = SteelWastewaterReuseSystem.mass_transfer_coefficient(
            self.feed_flow_rate, 
            self.channel_height, 
            1.0,  # length placeholder
            water_props['density'],
            water_props['viscosity'],
            avg_D
        )
        
        # Water flux (m/s)
        osmotic_pressure = SteelWastewaterReuseSystem.osmotic_pressure(total_conc_mol, self.temperature)
        self.water_flux = self.A * (self.transmembrane_pressure * 1e5 - osmotic_pressure)
        
        Jv = self.water_flux
        CP_modulus = np.exp(Jv / k) if k > 0 else 1.0
        
        # Effective flux considering CP
        self.effective_flux = self.water_flux / CP_modulus if CP_modulus > 0 else self.water_flux
        
        # Required membrane area (m2)
        if self.effective_flux > 0:
            self.required_area = self.permeate_flow_rate / self.effective_flux
        else:
            self.required_area = 0
        
        # Energy consumption (kW)
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / 
            (pump_efficiency * 1000)
        )
        
class ROUnit:
    """Reverse Osmosis unit with solution-diffusion model and concentration polarization"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        self.feed_pH = feed_pH
        
        self.solute_properties = SteelWastewaterReuseSystem.solute_properties
        # Membrane properties
        self.membrane = 'AFC99'
        self.membrane_thickness = 0.1e-6  # m (typical RO membrane)
        self.A_water = 2e-11  # Water permeability (m/s·Pa)
     
        # Operating conditions
        self.recovery_ratio = 0.85
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

    # CORREGIDO: Reemplazado con el solver iterativo robusto
    def solve(self):
        """Solve RO system using solution-diffusion model with CP"""
        water_props = SteelWastewaterReuseSystem.water_properties(self.temperature)
        
        # Flujos
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Diccionario de concentraciones de alimentación en mol/m³
        conc_mol_dict = SteelWastewaterReuseSystem.conc_mass_to_mol(
            self.feed_concentration, 
            self.solute_properties
        )
        
        # Calcular coeficientes de transferencia de masa (k) para todos los solutos
        k_dict = {}
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            D_value = self.solute_properties[comp_key].get('D', 1e-9)
            k_dict[comp_key] = SteelWastewaterReuseSystem.mass_transfer_coefficient(
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
            pi_m = SteelWastewaterReuseSystem.osmotic_pressure(sum(C_m_dict.values()), self.temperature)
            pi_p = SteelWastewaterReuseSystem.osmotic_pressure(sum(C_p_dict.values()), self.temperature)
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
            Mw = SteelWastewaterReuseSystem.solute_properties[comp_key]['Mw']
            
            # CORREGIDO: Conversión de C_p (mol/m³) a mg/L
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
        
class MDUnit:
    """Membrane Distillation unit with Dusty Gas Model"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate
        self.feed_concentration = feed_concentration
        self.feed_pH = feed_pH
        
        # Membrane properties
        self.membrane = 'PE1'
        self.membrane_thickness = 68e-6  # m
        self.pore_size = 0.1e-6  # m
        self.porosity = 0.6
        self.tortuosity = 1.5
        
        # Operating conditions
        self.recovery_ratio = 0.9
        self.feed_temp = 60  # °C
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
        feed_props = SteelWastewaterReuseSystem.water_properties(self.feed_temp)
        permeate_props = SteelWastewaterReuseSystem.water_properties(self.permeate_temp)
        
        # Vapor pressures (Pa)
        P_f = feed_props['vapor_pressure']
        P_p = permeate_props['vapor_pressure']
        
        # Mass transfer coefficient kg/m2sPa
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
            feed_conc = self.feed_concentration[comp] # Conc. de entrada a MD
            
            self.permeate_concentration[comp_key] = 0  # Ideal MD rejects all non-volatiles
            
            # CORREGIDO: Balance de masas robusto
            if self.concentrate_flow_rate > 0:
                self.concentrate_concentration[comp_key] = (
                    (feed_conc * self.feed_flow_rate - 
                     self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                    self.concentrate_flow_rate
                )
            else:
                self.concentrate_concentration[comp_key] = 0
        
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
        else:
            self.required_area = 0
        
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
        
        # Asegurarse de que los flujos estén definidos
        if self.diluate_flow_rate is None or self.concentrate_flow_rate is None:
             self.diluate_flow_rate = self.feed_flow_rate * self.recovery_ratio
             self.concentrate_flow_rate = self.feed_flow_rate * (1 - self.recovery_ratio)
             
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
        
        # 1. Flow Rates (si no se han calculado antes)
        if self.diluate_flow_rate is None:
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
        
   
        
class ChemicalPrecipitation:
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float], feed_pH: float):
        self.feed_flow_rate = feed_flow_rate  # m3/s
        self.feed_concentration = feed_concentration  # mg/L
        self.feed_pH = feed_pH
        

        self.coagulant_dose = 0.2  # kg/m3
        self.dosing_ratio = 1.5
        self.precipitant = 'NaOH'  # Primary precipitant
        self.coagulant = 'FeCl3'  # Coagulant aid
        self.flocculant_dose = 0.05  # kg/m3
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
                    if pH_range > 0:
                        removal_efficiency[metal] = 0.70 + (efficiency_range * (self.pH_target - props['min_pH']) / pH_range)
                    else:
                        removal_efficiency[metal] = 0.70
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
        
        coagulant_consumption = self.coagulant_dose * self.feed_flow_rate * 1e-3  # kg/s
        
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
            self.neutralizing_agent = 'HCl'  # Para aguas básicas
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
        
        info = {
            'feed_pH': self.feed_pH,
            'target_pH': self.target_pH,
            'neutralizing_agent': self.neutralizing_agent,
            'process_type': 'Acid neutralization' if self.feed_pH < 7.0 else 'Alkali neutralization',
            'net_acidity': acidity,
            'net_alkalinity': alkalinity
        }
        return info

        
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
    feed_pH = 9
    
    # Create and solve system
    system = SteelWastewaterReuseSystem(feed_flow, feed_conc, feed_pH)
    system.solve_system()
    performance = system.get_system_performance()
    
