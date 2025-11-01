"""
SEIR Model Simulation Engine
============================
Core epidemic simulation using stochastic agent-based approach
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

from .disease_params import DiseaseParameters, DiseaseDistributions
from .population import Population, Person, DiseaseState


@dataclass
class SimulationConfig:
    """Configuration for simulation run"""
    total_days: int = 365
    initial_infections: int = 10
    contact_rate_base: float = 12.0  # Average contacts per day
    dt: float = 1.0  # Time step (days)
    seed: Optional[int] = None


class SEIRSimulator:
    """
    Stochastic SEIR epidemic simulator
    Agent-based model tracking each individual
    """
    
    def __init__(self,
                 population: Population,
                 disease_params: DiseaseParameters,
                 config: SimulationConfig):
        """
        Initialize simulator
        
        Args:
            population: Population object
            disease_params: Disease parameter object
            config: Simulation configuration
        """
        self.population = population
        self.params = disease_params
        self.config = config
        self.distributions = DiseaseDistributions(disease_params, seed=config.seed)
        
        self.rng = np.random.default_rng(config.seed)
        
        # Tracking
        self.current_day = 0
        self.history = []
        
        # Contact rate modifications (for interventions - Phase 3)
        self.contact_rate_multiplier = 1.0
        
    def initialize_infections(self):
        """Seed initial infections randomly"""
        susceptible = self.population.get_susceptible()
        
        if len(susceptible) < self.config.initial_infections:
            n_initial = len(susceptible)
        else:
            n_initial = self.config.initial_infections
        
        # Randomly select initial infected
        initial_infected = self.rng.choice(
            susceptible,
            size=n_initial,
            replace=False
        )
        
        for person in initial_infected:
            self._expose_person(person)
            # Immediately transition to infected (skip exposed state for initial)
            person.time_in_state = person.incubation_time
            self._transition_E_to_I(person)
    
    def _expose_person(self, person: Person):
        """Expose a susceptible person"""
        person.state = DiseaseState.EXPOSED
        person.time_in_state = 0.0
        
        # Sample disease characteristics
        person.incubation_time = self.distributions.sample_incubation_period(1)[0]
        person.infectious_time = self.distributions.sample_infectious_period(1)[0]
        person.severity = self.distributions.sample_severity(1)[0]
        
        # Determine outcome
        person.will_die = self.distributions.sample_fatality(
            np.array([person.age]),
            np.array([person.severity])
        )[0]
    
    def _transition_E_to_I(self, person: Person):
        """Transition from Exposed to Infected"""
        person.state = DiseaseState.INFECTED
        person.time_in_state = 0.0
    
    def _transition_I_to_R_or_D(self, person: Person):
        """Transition from Infected to Recovered or Dead"""
        if person.will_die:
            person.state = DiseaseState.DEAD
        else:
            person.state = DiseaseState.RECOVERED
        person.time_in_state = 0.0
    
    def _calculate_transmission_rate(self, infected_person: Person) -> float:
        """
        Calculate effective transmission rate for an infected person
        Depends on: infectiousness curve, contact rate, interventions
        """
        # Infectiousness curve: peaks around day 3-5 of infection
        days_infected = infected_person.time_in_state
        
        # Simple peaked curve
        if days_infected < 0:
            infectiousness = 0.0
        elif days_infected < 3:
            infectiousness = days_infected / 3
        elif days_infected < 8:
            infectiousness = 1.0
        else:
            infectiousness = max(0, 1.0 - (days_infected - 8) / 4)
        
        # Base transmission rate
        beta = self.params.beta_base
        
        # Modify by asymptomatic status (asymptomatic less infectious)
        if infected_person.severity == 'asymptomatic':
            beta *= 0.5
        
        # Apply intervention multiplier
        beta *= self.contact_rate_multiplier
        
        return beta * infectiousness
    
    def _transmission_step(self):
        """
        Simulate transmission events for one time step
        Stochastic: each S-I pair has probability of transmission
        """
        susceptible = self.population.get_susceptible()
        infected = self.population.get_infected()
        
        if len(susceptible) == 0 or len(infected) == 0:
            return
        
        # For computational efficiency with large populations:
        # Sample contacts rather than checking all pairs
        
        for s_person in susceptible:
            # Number of contacts this person makes today
            contact_rate = self.config.contact_rate_base * self.contact_rate_multiplier
            n_contacts = self.rng.poisson(contact_rate)
            
            if n_contacts == 0:
                continue
            
            # Sample contacts from infected pool
            # (In reality, most contacts are with susceptible, but we only care about infected)
            # Probability of contacting an infected person
            p_infected = len(infected) / self.population.size
            n_infected_contacts = self.rng.binomial(n_contacts, p_infected)
            
            if n_infected_contacts == 0:
                continue
            
            # For each infected contact, check if transmission occurs
            for _ in range(n_infected_contacts):
                # Sample a random infected person
                infected_person = self.rng.choice(infected)
                
                # Transmission probability
                beta_eff = self._calculate_transmission_rate(infected_person)
                p_transmit = self.distributions.transmission_probability(
                    beta_eff,
                    contact_duration=1.0 / contact_rate  # Average duration per contact
                )
                
                # Transmission event
                if self.rng.random() < p_transmit:
                    self._expose_person(s_person)
                    break  # Person is now exposed, stop checking more contacts
    
    def _progression_step(self):
        """
        Update disease progression for all infected individuals
        E → I after incubation period
        I → R/D after infectious period
        """
        # Process exposed → infected
        for person in self.population.get_exposed():
            if person.time_in_state >= person.incubation_time:
                self._transition_E_to_I(person)
        
        # Process infected → recovered/dead
        for person in self.population.get_infected():
            if person.time_in_state >= person.infectious_time:
                self._transition_I_to_R_or_D(person)
    
    def _record_state(self):
        """Record current state for history"""
        counts = self.population.get_state_counts()
        
        record = {
            'day': self.current_day,
            'S': counts[DiseaseState.SUSCEPTIBLE],
            'E': counts[DiseaseState.EXPOSED],
            'I': counts[DiseaseState.INFECTED],
            'R': counts[DiseaseState.RECOVERED],
            'D': counts[DiseaseState.DEAD],
            'total': self.population.size
        }
        
        self.history.append(record)
    
    def step(self):
        """Execute one simulation time step"""
        # 1. Transmission (S → E)
        self._transmission_step()
        
        # 2. Disease progression (E → I, I → R/D)
        self._progression_step()
        
        # 3. Update time counters
        for person in self.population.people:
            if person.state in [DiseaseState.EXPOSED, DiseaseState.INFECTED]:
                person.time_in_state += self.config.dt
        
        # 4. Record state
        self._record_state()
        
        # 5. Increment day
        self.current_day += 1
    
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run full simulation
        
        Args:
            verbose: Print progress
            
        Returns:
            DataFrame with time series of SEIR states
        """
        # Initialize
        self.initialize_infections()
        self._record_state()
        
        if verbose:
            print(f"Starting simulation...")
            print(f"Population: {self.population.size}")
            print(f"Initial infections: {self.config.initial_infections}")
            print(f"Duration: {self.config.total_days} days")
            print(f"Estimated R0: {self.params.R0_estimate:.2f}")
            print()
        
        # Main simulation loop
        for day in range(self.config.total_days):
            self.step()
            
            if verbose and day % 30 == 0:
                counts = self.history[-1]
                print(f"Day {day:3d}: S={counts['S']:6d}, E={counts['E']:5d}, "
                      f"I={counts['I']:5d}, R={counts['R']:6d}, D={counts['D']:4d}")
        
        if verbose:
            final = self.history[-1]
            print(f"\nSimulation complete!")
            print(f"Total deaths: {final['D']}")
            print(f"Attack rate: {100 * (final['R'] + final['D']) / self.population.size:.1f}%")
        
        return pd.DataFrame(self.history)
    
    def get_results(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        return pd.DataFrame(self.history)


def plot_results(df: pd.DataFrame, title: str = "SEIR Epidemic Simulation"):
    """
    Plot SEIR curves
    
    Args:
        df: Results DataFrame from simulation
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: All states
    ax1.plot(df['day'], df['S'], label='Susceptible', color='blue', linewidth=2)
    ax1.plot(df['day'], df['E'], label='Exposed', color='orange', linewidth=2)
    ax1.plot(df['day'], df['I'], label='Infected', color='red', linewidth=2)
    ax1.plot(df['day'], df['R'], label='Recovered', color='green', linewidth=2)
    ax1.plot(df['day'], df['D'], label='Dead', color='black', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Number of People', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Active cases and daily deaths
    ax2_twin = ax2.twinx()
    
    ax2.plot(df['day'], df['I'], label='Active Infections', color='red', linewidth=2)
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_ylabel('Active Infections', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3)
    
    # Daily deaths (derivative)
    daily_deaths = df['D'].diff().fillna(0)
    ax2_twin.plot(df['day'], daily_deaths, label='Daily Deaths', 
                  color='black', linewidth=2, linestyle='--')
    ax2_twin.set_ylabel('Daily Deaths', fontsize=12, color='black')
    ax2_twin.tick_params(axis='y', labelcolor='black')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example simulation
    print("SEIR Model Test Run")
    print("=" * 60)
    
    # Create population
    pop = Population(size=50000, seed=42)
    
    # Set up disease parameters
    params = DiseaseParameters()
    
    # Configure simulation
    config = SimulationConfig(
        total_days=365,
        initial_infections=10,
        seed=42
    )
    
    # Run simulation
    simulator = SEIRSimulator(pop, params, config)
    results = simulator.run(verbose=True)
    
    # Plot results
    import matplotlib.pyplot as plt
    fig = plot_results(results)
    plt.savefig('seir_basic_simulation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'seir_basic_simulation.png'")
