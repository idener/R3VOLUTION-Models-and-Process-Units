# -*- coding: utf-8 -*-
"""
Enhanced membrane process model with detailed physical equations for NF, RO, and MD processes
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.constants import R, calorie

class MembraneProcessModel:
    """
    Comprehensive membrane process model integrating NF, RO, and MD with detailed physical modeling.
    Includes:
    - Solution-diffusion model for RO/NF
    - Dusty gas model for MD
    - Extended Nernst-Planck for charged species
    - Heat and mass transfer correlations
    - Detailed cost models
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
            '2-MeTHF': {'Mw': 86.13, 'D': 1.0e-9, 'charge': 0, 'Stokes_radius': 0.3e-9,'B': 1e-8},
            'Xylose': {'Mw': 150.13, 'D': 0.67e-9, 'charge': 0, 'Stokes_radius': 0.45e-9,'B': 5e-9},
            'Lignin': {'Mw': 10000, 'D': 0.01e-9, 'charge': -1, 'Stokes_radius': 2.0e-9,'B': 1e-9},
            'Na': {'Mw': 23, 'D': 1.33e-9, 'charge': +1, 'Stokes_radius': 0.18e-9, 'B': 2e-8},
            'Glycoxyllic_acid': {'Mw': 104, 'D': 0.9e-9, 'charge': -1, 'Stokes_radius': 0.35e-9, 'B': 3e-9},
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
    
    def conc_mass_to_mol(feed_conc_dict: Dict[str, float], solute_properties) -> Dict[str, float]:

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
        self.nf = NFUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
        
        self.ro = ROUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )
        
        self.md = MDUnit(
            feed_flow_rate=None,
            feed_concentration={}
        )

    def solve_system(self):
        """Solve material and energy balances for the entire system with detailed physics"""
        # Solve each unit in sequence
        self.nf.feed_flow_rate = self.feed_flow_rate
        self.nf.feed_concentration = self.feed_concentration
        self.nf.solve()
        
        self.ro.feed_flow_rate = self.nf.permeate_flow_rate
        self.ro.feed_concentration = self.nf.permeate_concentration
        self.ro.solve()
        
        # Mix NF and RO concentrates for MD feed
        self.md.feed_flow_rate = self.ro.concentrate_flow_rate + self.nf.concentrate_flow_rate
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')  # Handle space in 'Glycoxyllic acid'
            nf_flow = self.nf.concentrate_flow_rate
            ro_flow = self.ro.concentrate_flow_rate
            nf_conc = self.nf.concentrate_concentration.get(comp_key, 0)
            ro_conc = self.ro.concentrate_concentration.get(comp_key, 0)
            
            self.md.feed_concentration[comp_key] = (
                (ro_conc * ro_flow + nf_conc * nf_flow) / 
                self.md.feed_flow_rate if self.md.feed_flow_rate > 0 else 0
            )
        
        self.md.solve()

    # ... (rest of the class methods remain the same)
    def get_system_performance(self) -> Dict:
         """Calculate overall system performance metrics"""
       
         
         water_recovery = (self.md.permeate_flow_rate + self.ro.permeate_flow_rate) / self.feed_flow_rate
         
         contaminants_removal = {
             '2-MeTHF': 1 - (((self.md.permeate_concentration['2-MeTHF']*self.md.permeate_flow_rate) + (self.ro.permeate_concentration['2-MeTHF'] * self.ro.permeate_flow_rate)) / (self.feed_concentration['2-MeTHF']*self.feed_flow_rate)),
             'Xylose':  1 - (((self.md.permeate_concentration['Xylose']*self.md.permeate_flow_rate) + (self.ro.permeate_concentration['Xylose'] * self.ro.permeate_flow_rate)) / (self.feed_concentration['Xylose']*self.feed_flow_rate)),
             'Lignin': 1 - (((self.md.permeate_concentration['Lignin']*self.md.permeate_flow_rate) + (self.ro.permeate_concentration['Lignin'] * self.ro.permeate_flow_rate)) / (self.feed_concentration['Lignin']*self.feed_flow_rate)),
             'Na':  1 - (((self.md.permeate_concentration['Na']*self.md.permeate_flow_rate) + (self.ro.permeate_concentration['Na'] * self.ro.permeate_flow_rate)) / (self.feed_concentration['Na']*self.feed_flow_rate)),
             'Glycoxyllic acid': 1 - (((self.md.permeate_concentration['Glycoxyllic_acid']*self.md.permeate_flow_rate) + (self.ro.permeate_concentration['Glycoxyllic_acid'] * self.ro.permeate_flow_rate)) / (self.feed_concentration['Glycoxyllic_acid']*self.feed_flow_rate)),
         }
         
         electricity_consumption={'NF':self.nf.energy_consumption,
                                  'RO':self.ro.energy_consumption,
                                  'MD':self.md.electrical_energy
                                  }
         
         Flows={'Alimentación NF':feed_flow_rate,
                 'Alimentación RO': self.ro.feed_flow_rate, #Same flow as NF permeate
                 'Alimentación MD': self.md.feed_flow_rate, #Same flow as RO permeate
                 'Salida Proceso': self.md.permeate_flow_rate+self.ro.permeate_flow_rate,
                 'Concentrado': self.md.concentrate_flow_rate
                 }
         
         membranes_areas_m2 = {'nf': self.nf.required_area,
                                 'ro': self.ro.required_area,
                                 'md': self.md.required_area}
         
         return {'contaminants_removal': contaminants_removal,
                 'recycled_water_flow': self.md.permeate_flow_rate + self.ro.permeate_flow_rate,
                 'water_recovery': water_recovery,
                 'outlet_concentrations': self.md.permeate_concentration,
                 'electricity_consumption_kW': electricity_consumption,
                 'flows_m3/s':Flows,
                 'heat_consumption_kW':self.md.thermal_energy/1000,
                 'membrane_areas': membranes_areas_m2
                 }
     
        
    def operational_cost(self): 
         # System-wide parameters
         self.operational_hours_per_year = 8000  # Typical operational hours
         self.electricity_cost = 0.1  # $/kWh
         self.heat_cost = 0.05 #$/kWh
         self.water_value = 2.5  # $/m3 value of recycled water
         
         self.year_electricity_cost=self.electricity_cost*(self.nf.energy_consumption+self.ro.energy_consumption)*self.operational_hours_per_year
         self.year_heat_cost=self.md.thermal_energy*self.heat_cost*self.operational_hours_per_year 
         
         self.savings=(self.ro.permeate_flow_rate+self.md.permeate_flow_rate_TA)*self.operational_hours_per_year*self.water_value*3600 #$/year
         
         self.cost=self.year_electricity_cost+self.year_heat_cost #$/year
         self.cash_flow=self.savings-self.cost #$/year
         return self.cash_flow
     
    def invest_cost (self): 
        
        self.NF_membrane_cost=self.nf.membrane_cost * self.nf.required_area
        self.RO_membrane_cost=self.ro.membrane_cost * self.ro.required_area
        self.MD_membrane_cost=self.md.membrane_cost * self.md.required_area
        
        
        def pump_cost(P:float, W_pump:float):
            
            K = [3.3892, 0.0538, 0.1538]
            C = [(-1) * 0.3935, 0.3957, (-1) * 0.00226]
            B = [1.89, 1.35]
            Fm= 2.3#Stainless Steel
            n= 0.6
            Win=1 if W_pump <1 else 300 if W_pump > 300 else W_pump
            
            fp=10 ** (C[0] + C[1] * np.log10(P) + C[2] * np.log10(P) ** 2)
            Cpo=10**(K[0] + K[1] * np.log10(Win) + K[2] * np.log10(Win) ** 2)
            
            Co=((W_pump/Win)**n)*Cpo
            Cp = Co * (B[0] + B[1] * fp * Fm)
            
            return Cp
        
        self.NF_pump_cost=pump_cost(self.nf.transmembrane_pressure,self.nf.energy_consumption)
        self.RO_pump_cost=pump_cost(self.ro.transmembrane_pressure,self.ro.energy_consumption)
        self.MD_pump_cost=pump_cost(2,self.md.electrical_energy)
       
        
        
        self.NF_invest_cost=self.NF_membrane_cost + self.NF_pump_cost
        self.RO_invest_cost=self.RO_membrane_cost + self.RO_pump_cost
        self.MD_invest_cost=self.MD_membrane_cost + self.MD_pump_cost
         
        self.capital_cost=self.NF_invest_cost+self.RO_invest_cost+self.MD_invest_cost
        return self.capital_cost
    
    def NPV (self):
        discount_rate=0.1
        lifetime_years=20
        self.net_present_value=-self.invest_cost()
        for year in range(1, lifetime_years + 1):
          self.net_present_value += self.operational_cost() / (1 + discount_rate) ** year
        return self.net_present_value
    
    def _npv_wrapper(self, x):
        # Update instance attributes based on the optimization variables
        self.nf.recovery_ratio, self.ro.recovery_ratio, self.md.recovery_ratio, \
        self.nf.flux, self.ro.flux, self.md.feed_temp = x
        # Log the current values and the corresponding NPV
        system.solve_system()
        npv_value = self.NPV()
        print(f"Current x: {x}, NPV: {npv_value}")
        # Call the NPV method which uses the updated instance attributes
        return -npv_value
    
    def optimization (self):
        x = [self.nf.recovery_ratio, self.ro.recovery_ratio, self.md.recovery_ratio,self.nf.flux,self.ro.flux,self.md.feed_temp]
        bounds=[(0.1,0.9),(0.1,0.9),(0.1,0.5),(3,7),(3,8),(65,100)]
            
        result=opt.minimize(self._npv_wrapper,x,bounds=bounds,method='Powell')
            
        return result  
        
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
        self.recovery_ratio = 0.81
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
            
            # CORREGIDO: Manejar exclusión estérica total
            if lambda_ >= 1:
                S = 0.0  
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
        
        # CORREGIDO: Obtener diccionario de concentraciones molares
        conc_mol_dict = MembraneProcessModel.conc_mass_to_mol(self.feed_concentration, MembraneProcessModel.solute_properties)
        
        # CORREGIDO: Calcular presión osmótica TOTAL
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
        # CORREGIDO: Usar presión osmótica total
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

    def solve(self):
        """
        CORREGIDO: Solve RO system using solution-diffusion model with CP
        La lógica ha sido reestructurada para calcular Jw (flujo de agua)
        basándose en la presión osmótica TOTAL, de forma iterativa.
        """
        water_props = MembraneProcessModel.water_properties(self.temperature)
        
        self.permeate_flow_rate = self.feed_flow_rate * self.recovery_ratio
        self.concentrate_flow_rate = self.feed_flow_rate - self.permeate_flow_rate
        
        # Diccionario de concentraciones de alimentación en mol/m³
        conc_mol_dict = MembraneProcessModel.conc_mass_to_mol(self.feed_concentration, MembraneProcessModel.solute_properties)
        
        # Calcular coeficientes de transferencia de masa (k) para todos los solutos
        k_dict = {}
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            k_dict[comp_key] = MembraneProcessModel.mass_transfer_coefficient(
                self.feed_flow_rate,
                self.channel_height,
                1.0,  # length placeholder
                water_props['density'],
                water_props['viscosity'],
                self.solute_properties[comp_key].get('D', 1e-9)
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
                C_m_dict[comp_key] = feed_conc_mol * np.exp(self.Jw / k)
            
            # 2. Calcular C_p (permeado) para todos los solutos
            for comp_key in C_p_dict:
                B = self.solute_properties[comp_key].get('B', 1e-9) # Permeabilidad del soluto
                Js = B * (C_m_dict[comp_key] - C_p_dict[comp_key]) # Flujo de soluto
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
        

        # Guardar concentraciones finales
        for comp in self.feed_concentration.keys():
            comp_key = comp.replace(' ', '_')
            Mw = MembraneProcessModel.solute_properties[comp_key]['Mw']
            
            # CORREGIDO: Conversión de C_p (mol/m³) a mg/L
            # (mol/m³) * (g/mol) = g/m³ 
            # g/m³ es numéricamente igual a mg/L
            self.permeate_concentration[comp_key] = C_p_dict[comp_key] * Mw
            
            # Calcular concentración del concentrado
            feed_conc_mgL = self.feed_concentration[comp]
            self.concentrate_concentration[comp_key] = (
                (feed_conc_mgL * self.feed_flow_rate - 
                 self.permeate_concentration[comp_key] * self.permeate_flow_rate) / 
                self.concentrate_flow_rate if self.concentrate_flow_rate > 0 else 0
            )
            
        # Calcular área de membrana requerida
        self.required_area = self.permeate_flow_rate / self.Jw
        
        # Consumo de energía
        pump_efficiency = 0.7
        self.energy_consumption = (
            self.feed_flow_rate * self.transmembrane_pressure * 1e5 / (pump_efficiency * 1000)
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
        self.recovery_ratio = 0.72
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
        
        # Concentration calculations (assuming high rejection of non-volatiles)
        for comp_key in self.feed_concentration.keys():
            # comp_key ya tiene el '_' por cómo se construye en solve_system
            self.permeate_concentration[comp_key] = 0.05 * self.feed_concentration[comp_key]  # Ideal MD rejects all non-volatiles 
            "Waiting for aditional data, but elimination will be higher than 90% in MD"
            
            # CORREGIDO: Balance de masas del concentrado
            self.concentrate_concentration[comp_key] = (
                (self.feed_concentration[comp_key] * self.feed_flow_rate - 
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
            
if __name__ == "__main__":
    # Define feed characteristics 
    feed_flow_rate =10/3600  # m3/s (1 m3/h)
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
    performance=system.get_system_performance()