# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.constants import R, calorie

class MembraneProcessModel:
    """
    Comprehensive membrane process model integrating NF, RO, and MD with detailed physical modeling.
    Includes:
    1. Microfiltration (MF) - concentrates pulp fibers & inorganic compounds
    2. Ultrafiltration (UF) - concentrates pigments
    3. Nanofiltration (NF) - concentrates organic materials (regulators, brighteners, biocides)
    4. Membrane Distillation (MD) - produces clean water and concentrated brine
    """
    
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
    
    def conc_mass_to_mol(feed_conc_dict: Dict[str, float], solute_properties: Dict) -> Dict[str, float]:
        """

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
        self.mf = MFUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
        
        self.uf = UFUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
        
        self.nf = NFUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
        
        self.md = MDUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
        
    def solve_system(self):
        # Solve each unit in sequence
        self.mf.feed_flow_rate = self.feed_flow_rate
        self.mf.feed_concentration = self.feed_concentration
        self.mf.solve()
        
        self.uf.feed_flow_rate = self.mf.permeate_flow_rate
        self.uf.feed_concentration = self.mf.permeate_concentration
        self.uf.solve()
        
        self.nf.feed_flow_rate = self.uf.permeate_flow_rate
        self.nf.feed_concentration = self.uf.permeate_concentration
        self.nf.solve()
     
        self.md.feed_flow_rate = self.nf.permeate_flow_rate
        self.md.feed_concentration = self.nf.permeate_concentration
        self.md.solve()

    def get_system_performance(self) -> Dict:
         """Calculate overall system performance metrics"""
         water_recovery = self.md.permeate_flow_rate / self.feed_flow_rate * 100
         contaminants_removal = {'fibers_inorganics': (1- self.md.permeate_concentration['fibers_inorganics']*self.md.permeate_flow_rate /self.feed_concentration['fibers_inorganics']*self.feed_flow_rate) * 100,
                                'pigments': (1 - self.md.permeate_concentration['pigments']*self.md.permeate_flow_rate /self.feed_concentration['pigments']*self.feed_flow_rate) * 100,
                                'organics': (1- self.md.permeate_concentration['organics']*self.md.permeate_flow_rate /self.feed_concentration['organics']*self.feed_flow_rate) * 100,
                                'brine': (1 - self.md.permeate_concentration['brine']*self.md.permeate_flow_rate /self.feed_concentration['brine']*self.feed_flow_rate) * 100
                                }
         
         electricity_consumption = {'MF': self.mf.energy_consumption,
                                    'UF': self.uf.energy_consumption,
                                    'NF': self.nf.energy_consumption,
                                    'MD': self.md.electrical_energy
                                    }
         
         membrane_areas_m2 = {'MF': self.mf.required_area,
                              'UF': self.uf.required_area,
                              'NF': self.nf.required_area,
                              'MD': self.md.required_area
                              }
         heat_consumption = self.md.thermal_energy
         
         return {'Contaminants removal (%)': contaminants_removal,
                 'Recycled water flow (m3/h)': self.md.permeate_flow_rate*3600,
                 'Membrane areas (m2)': membrane_areas_m2,
                 'Water recovery (%)': water_recovery,
                 'Electricity consumption (kW)': electricity_consumption,
                 'Heat consumption (kW)': heat_consumption
                 }
class MFUnit:
    """Microfiltration unit with pore flow model (simplified without cake dynamics)"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float]):
        self.feed_flow_rate = feed_flow_rate  # m3/s
        self.feed_concentration = feed_concentration  # mg/L
        
        # Membrane properties
        self.membrane = 'MF-100'
        self.membrane_thickness = 150e-6  # m
        self.pore_radius = 0.1e-6  # m (100 nm)
        self.porosity = 0.7
        self.tortuosity = 1.5
        self.membrane_resistance = 1e11  # m⁻¹ (solo resistencia de membrana)
        
        # Operating conditions
        self.recovery_ratio = 0.9
        self.transmembrane_pressure = 1.0  # bar
        self.temperature = 20  # °C
        self.cross_flow_velocity = 1.0  # m/s
        self.channel_height = 2e-3  # m
        
        # Initialize calculated properties
        self.permeate_flow_rate = None
        self.concentrate_flow_rate = None
        self.permeate_concentration = {}
        self.concentrate_concentration = {}
        self.energy_consumption = None
        self.required_area = None
        self.water_flux = None
        
        
        self.solute_properties = MembraneProcessModel.solute_properties
    def calculate_rejection(self):
        """Simplified MF rejection model based solely on size exclusion"""
        rejection = {}
        
        for comp, props in self.solute_properties.items():
            # Simple size exclusion
            lambda_ = props['Stokes_radius'] / self.pore_radius
            # Esta lógica es correcta: si lambda_ >= 1, el rechazo es 1.0 (100%)
            rejection[comp] = 1 - (1 - lambda_)**2 if lambda_ < 1 else 1.0
                
        return rejection
    
    def calculate_flux(self):
        """Simple Darcy's law with only membrane resistance"""
        water_props = MembraneProcessModel.water_properties(self.temperature)
        self.water_flux = (self.transmembrane_pressure * 1e5) / (water_props['viscosity'] * self.membrane_resistance)
        return self.water_flux
    
    def solve(self):
        """Simplified solution without fouling dynamics"""
        water_props = MembraneProcessModel.water_properties(self.temperature)
        
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
            
            # Este balance de masas es correcto
            self.concentrate_concentration[comp_key] = (
                (feed_conc * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
        
        # Mass transfer coefficient (m/s) - Calculado pero no usado, consistente con "simplified"
        k = 0.023 * (self.cross_flow_velocity**0.8) * (water_props['viscosity']**-0.47) * (1e-10**0.33)
        
        # Required membrane area (m²)
        self.required_area = self.permeate_flow_rate / self.water_flux
        
        # Energy consumption (kW)
        pump_efficiency = 0.65
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / 
            (pump_efficiency * 1000)
        )    
    
    
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
        self.A = 1e-10  # Water permeability (m/s·Pa)
        
        # Operating conditions
        self.recovery_ratio = 0.9
        self.transmembrane_pressure = 10
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
        water_props = MembraneProcessModel.water_properties(self.temperature)
        
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
            
            # Este balance de masas es correcto
            self.concentrate_concentration[comp_key] = (
                (feed_conc * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
        
        # Required membrane area (m²)
        self.required_area = self.permeate_flow_rate / self.water_flux
        
        # Energy consumption (kW)
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / 
            (pump_efficiency * 1000)
        )


class NFUnit:
    """Nanofiltration unit with solution-diffusion and Donnan exclusion models"""
    
    def __init__(self, feed_flow_rate: float, feed_concentration: Dict[str, float]):
        self.feed_flow_rate = feed_flow_rate  # m3/s
        self.feed_concentration = feed_concentration  # mg/L
        
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
        self.A=1e-10 # m/s·Pa Water permeability
        
    def calculate_rejection(self):
        """
        Calculate rejection coefficients using extended Nernst-Planck with Donnan exclusion
        for charged species and steric hindrance for neutral species
        """
        #water_props = MembraneProcessModel.water_properties(self.temperature)
        rejection = {}
        self.solute_properties=MembraneProcessModel.solute_properties
        for comp, props in self.solute_properties.items():
            # Steric hindrance parameter
            lambda_ = props['Stokes_radius'] / self.pore_radius
            

            if lambda_ >= 1:
                S = 0.0  # Exclusión total, el soluto no entra al poro
            else:
                S = (1 - lambda_)**2  # Factor estérico
            
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
                rejection[comp] = 1 - (B / (self.A * self.transmembrane_pressure * 1e5))
        
        return rejection
    
    def solve(self):
        """Solve material and energy balances with detailed transport models"""
        # Get water properties
        water_props = MembraneProcessModel.water_properties(self.temperature)
        
        # CORREGIDO: Llamar a la función arreglada que devuelve un diccionario
        conc_mol_dict = MembraneProcessModel.conc_mass_to_mol(self.feed_concentration, MembraneProcessModel.solute_properties)
        
        # CORREGIDO: Calcular la presión osmótica TOTAL sumando todas las especies
        total_conc_mol = sum(conc_mol_dict.values())
        total_osmotic_pressure = MembraneProcessModel.osmotic_pressure(total_conc_mol, self.temperature)

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
            self.concentrate_concentration[comp_key] = (
                (feed_conc * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
            
        # Mass transfer coefficient (m/s)
        k = MembraneProcessModel.mass_transfer_coefficient(
            self.feed_flow_rate, 
            self.channel_height, 
            1.0,  # length placeholder
            water_props['density'],
            water_props['viscosity'],
            1e-9  # diffusion coefficient placeholder
        )
        
        # Water flux (m/s)

        self.water_flux = (self.A * (self.transmembrane_pressure * 1e5 - total_osmotic_pressure))
        
        Jv = self.water_flux
        CP_modulus = np.exp(Jv / k)
        
        # Effective flux considering CP
        self.effective_flux = self.water_flux / CP_modulus 
        
        # Required membrane area (m2)
        self.required_area = self.permeate_flow_rate / self.effective_flux
        
        # Energy consumption (kW)
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / 
            (pump_efficiency * 1000)
        )
        
        
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
        self.recovery_ratio = 0.9
        self.feed_temp = 70 # °C
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
        
    def solve(self):
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
            feed_conc = self.feed_concentration[comp_key] # Usar comp_key o comp está bien aquí
            
            self.permeate_concentration[comp_key] = 0.05 * feed_conc  # Ideal MD rejects all non-volatiles
            "Waiting for aditional data, but elimination will be higher than 90% in MD"
            
            # CORREGIDO: Balance de masas del concentrado
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
        if not self.required_area:
            self.required_area = (self.permeate_flow_rate / self.J) 
            
# Example usage
if __name__ == "__main__":
    # Define feed characteristics (typical pulp & paper mill wastewater)
    feed_flow_rate = 40/3600  # m3/s (360 m3/h)
    feed_concentration = {
        'fibers_inorganics': 5000,  # mg/L
        'pigments': 1000,  # mg/L
        'organics': 500,  # mg/L (regulators, brighteners, biocides)
        'brine': 2000  # mg/L (dissolved salts)
    }
    
    
    system = MembraneProcessModel(feed_flow_rate=feed_flow_rate, feed_composition=feed_concentration)
    system.solve_system()
    performance=system.get_system_performance()