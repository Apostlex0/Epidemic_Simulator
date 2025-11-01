"""
Disease Parameters and Distributions
====================================
Defines all epidemiological parameters with realistic probability distributions
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class DiseaseParameters:
    """Core disease parameters for SEIR+ model"""
    
    # Transmission
    beta_base: float = 0.4  # Base transmission rate (contacts per day × prob per contact)
    
    # Disease progression timings (in days)
    incubation_mean: float = 5.5  # Mean incubation period
    incubation_std: float = 2.3   # Std dev of incubation
    infectious_shape: float = 4.0  # Gamma shape for infectious period
    infectious_scale: float = 2.0  # Gamma scale (mean = shape × scale = 8 days)
    
    # Age-stratified Infection Fatality Rates (IFR)
    ifr_by_age: Dict[str, float] = None
    
    # Symptom severity probabilities
    severity_probs: Dict[str, float] = None
    
    # Contact rates by age/occupation
    contact_rates: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default distributions"""
        if self.ifr_by_age is None:
            self.ifr_by_age = {
                '0-9': 0.00002,
                '10-19': 0.00006,
                '20-29': 0.0003,
                '30-39': 0.0008,
                '40-49': 0.0015,
                '50-59': 0.006,
                '60-69': 0.02,
                '70-79': 0.05,
                '80+': 0.15
            }
        
        if self.severity_probs is None:
            self.severity_probs = {
                'asymptomatic': 0.35,
                'mild': 0.45,
                'severe': 0.15,
                'critical': 0.05
            }
        
        if self.contact_rates is None:
            self.contact_rates = {
                'child': 15,      # School-age children
                'adult': 12,      # Working adults
                'elderly': 8,     # Retired/elderly
                'healthcare': 20  # Healthcare workers
            }
    
    @property
    def R0_estimate(self) -> float:
        """Estimate basic reproduction number"""
        avg_infectious_period = self.infectious_shape * self.infectious_scale
        return self.beta_base * avg_infectious_period
    
    def get_ifr(self, age: int) -> float:
        """Get IFR for specific age"""
        if age < 10:
            return self.ifr_by_age['0-9']
        elif age < 20:
            return self.ifr_by_age['10-19']
        elif age < 30:
            return self.ifr_by_age['20-29']
        elif age < 40:
            return self.ifr_by_age['30-39']
        elif age < 50:
            return self.ifr_by_age['40-49']
        elif age < 60:
            return self.ifr_by_age['50-59']
        elif age < 70:
            return self.ifr_by_age['60-69']
        elif age < 80:
            return self.ifr_by_age['70-79']
        else:
            return self.ifr_by_age['80+']


class DiseaseDistributions:
    """
    Random variable generators for disease characteristics
    Uses realistic probability distributions from COVID-19 data
    """
    
    def __init__(self, params: DiseaseParameters, seed: int = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
    
    def sample_incubation_period(self, n: int = 1) -> np.ndarray:
        """
        Sample incubation periods (E → I transition time)
        Distribution: Lognormal(μ=5.5, σ=2.3 days)
        """
        mu = np.log(self.params.incubation_mean**2 / 
                    np.sqrt(self.params.incubation_mean**2 + self.params.incubation_std**2))
        sigma = np.sqrt(np.log(1 + (self.params.incubation_std / self.params.incubation_mean)**2))
        
        return self.rng.lognormal(mu, sigma, n)
    
    def sample_infectious_period(self, n: int = 1) -> np.ndarray:
        """
        Sample infectious periods (I → R transition time)
        Distribution: Gamma(shape=4, scale=2) → mean=8 days
        """
        return self.rng.gamma(
            self.params.infectious_shape,
            self.params.infectious_scale,
            n
        )
    
    def sample_severity(self, n: int = 1) -> np.ndarray:
        """
        Sample symptom severity
        Returns: array of strings ['asymptomatic', 'mild', 'severe', 'critical']
        """
        severities = list(self.params.severity_probs.keys())
        probs = list(self.params.severity_probs.values())
        
        return self.rng.choice(severities, size=n, p=probs)
    
    def sample_fatality(self, ages: np.ndarray, severities: np.ndarray) -> np.ndarray:
        """
        Sample whether infection results in death
        P(death) depends on age and severity
        """
        n = len(ages)
        death_probs = np.zeros(n)
        
        for i in range(n):
            base_ifr = self.params.get_ifr(ages[i])
            
            # Modify by severity
            severity_mult = {
                'asymptomatic': 0.01,
                'mild': 0.1,
                'severe': 1.0,
                'critical': 5.0
            }
            
            death_probs[i] = base_ifr * severity_mult.get(severities[i], 1.0)
            death_probs[i] = min(death_probs[i], 1.0)  # Cap at 100%
        
        return self.rng.random(n) < death_probs
    
    def transmission_probability(self, 
                                 beta: float,
                                 contact_duration: float = 1.0,
                                 distance: float = 1.0,
                                 masks: bool = False,
                                 ventilation: float = 1.0) -> float:
        """
        Calculate transmission probability for a single contact
        P(transmission) = 1 - exp(-β × duration × modifiers)
        """
        # Distance decay
        d0 = 2.0  # Characteristic distance (meters)
        distance_factor = np.exp(-distance / d0)
        
        # Mask reduction
        mask_factor = 0.3 if masks else 1.0
        
        # Combined effect
        effective_beta = beta * distance_factor * mask_factor * ventilation
        
        # Probability from exponential contact process
        return 1 - np.exp(-effective_beta * contact_duration)


# Default parameters instance
DEFAULT_PARAMS = DiseaseParameters()


if __name__ == "__main__":
    # Test the distributions
    params = DiseaseParameters()
    dist = DiseaseDistributions(params, seed=42)
    
    print("Disease Parameters Test")
    print("=" * 50)
    print(f"Estimated R0: {params.R0_estimate:.2f}")
    print(f"\nIncubation periods (5 samples): {dist.sample_incubation_period(5)}")
    print(f"Infectious periods (5 samples): {dist.sample_infectious_period(5)}")
    print(f"Severities (10 samples): {dist.sample_severity(10)}")
    
    # Test age-specific IFR
    ages = np.array([25, 45, 65, 85])
    print(f"\nIFR by age:")
    for age in ages:
        print(f"  Age {age}: {params.get_ifr(age):.6f}")
