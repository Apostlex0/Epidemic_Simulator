"""
Spatial SEIR Epidemic Simulator
================================
Combines SEIR dynamics with spatial structure and mobility
This is the integration of Phase 1 + Phase 2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .grid import SpatialGrid, GridCell
from .distance_kernel import DistanceKernel
from .mobility import MobilitySimulator, MovementConfig
from .realistic_interventions import RealisticInterventionManager, RealisticInterventionConfig
from ..core.disease_params import DiseaseParameters, DiseaseDistributions


@dataclass
class SpatialSimulationConfig:
    """Configuration for spatial epidemic simulation"""
    total_days: int = 365
    dt: float = 1.0
    
    # Grid parameters
    grid_rows: int = 20
    grid_cols: int = 20
    total_population: int = 1_000_000
    population_distribution: str = 'lognormal'  # 'uniform', 'lognormal', 'clustered'
    
    # Initial conditions
    initial_infection_cell: Tuple[int, int] = None  # None = random urban center
    initial_infections: int = 10
    
    # Mobility
    infected_travel_rate: float = 0.3   # Higher travel rate - many infected don't know they're sick
    
    # Interventions
    enable_interventions: bool = True
    
    # Random seed
    seed: int = None


class SpatialSEIRSimulator:
    """
    Spatial stochastic SEIR simulator
    Compartmental model on grid with mobility
    """
    
    def __init__(self,
                 config: SpatialSimulationConfig,
                 disease_params: DiseaseParameters = None,
                 movement_config: MovementConfig = None):
        """
        Initialize spatial simulator
        
        Args:
            config: Simulation configuration
            disease_params: Disease parameters (uses defaults if None)
            movement_config: Movement configuration (uses defaults if None)
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Disease parameters
        if disease_params is None:
            disease_params = DiseaseParameters()
        self.disease_params = disease_params
        self.distributions = DiseaseDistributions(disease_params, seed=config.seed)
        
        # Create spatial grid
        self.grid = SpatialGrid(
            rows=config.grid_rows,
            cols=config.grid_cols,
            total_population=config.total_population,
            population_distribution=config.population_distribution,
            seed=config.seed
        )
        
        # Create mobility model
        if movement_config is None:
            movement_config = MovementConfig()
        self.mobility = MobilitySimulator(self.grid, movement_config, seed=config.seed)
        
        # Distance kernel for transmission
        self.kernel = DistanceKernel(seed=config.seed)
        
        # Realistic intervention manager
        if config.enable_interventions:
            intervention_config = RealisticInterventionConfig()
            self.interventions = RealisticInterventionManager(
                intervention_config, 
                config.total_population
            )
        else:
            self.interventions = None
        
        # Tracking
        self.current_day = 0
        self.history = []
        
        # Cell-level tracking (for detailed spatial analysis)
        self.cell_history = []
    
    def initialize_epidemic(self):
        """Seed initial infections"""
        # Choose infection origin
        if self.config.initial_infection_cell is not None:
            origin_x, origin_y = self.config.initial_infection_cell
        else:
            # Choose most populated cell (urban center)
            max_pop = 0
            origin_x, origin_y = 0, 0
            
            for (x, y), cell in self.grid.cells.items():
                if cell.population > max_pop:
                    max_pop = cell.population
                    origin_x, origin_y = x, y
        
        # Seed infections
        self.grid.seed_infection(origin_x, origin_y, self.config.initial_infections)
        
        # Note: Initially placing in I compartment (can modify to use E if needed)
        print(f"Epidemic origin: Cell ({origin_x}, {origin_y}) with {self.grid.get_cell(origin_x, origin_y).population:,} people")
    
    def transmission_within_cell(self, cell: GridCell, intervention_effects: Dict[str, float] = None):
        """
        Simulate transmission within a single cell
        Uses well-mixed assumption within cell
        """
        if cell.S <= 0 or cell.I <= 0:
            return
        
        if intervention_effects is None:
            intervention_effects = {'contact_reduction': 1.0, 'transmission_reduction': 1.0}
        
        # Force of infection (FOI)
        prevalence = cell.I / (cell.S + cell.E + cell.I + cell.R)
        
        # Transmission rate adjusted by density
        # Higher density = more contacts
        density_factor = 1.0 + 0.1 * np.log10(cell.density + 1)
        beta_eff = self.disease_params.beta_base * density_factor
        
        # Apply intervention effects
        beta_eff *= intervention_effects['contact_reduction']      # Contact reduction (lockdown/behavior)
        beta_eff *= intervention_effects['transmission_reduction'] # Transmission reduction (masks)
        
        # Expected number of new infections
        lambda_param = beta_eff * cell.S * prevalence
        
        # Stochastic realization
        new_infections = self.rng.poisson(lambda_param)
        new_infections = min(new_infections, cell.S)  # Can't infect more than susceptible
        
        # Update compartments (S → E)
        cell.S -= new_infections
        cell.E += new_infections
    
    def transmission_between_cells(self, cell1: GridCell, cell2: GridCell, distance: float, intervention_effects: Dict[str, float] = None):
        """
        Simulate transmission between neighboring cells
        Reduced by distance
        """
        if cell1.I <= 0 or cell2.S <= 0:
            return
        
        if intervention_effects is None:
            intervention_effects = {'contact_reduction': 1.0, 'transmission_reduction': 1.0}
        
        # Distance-dependent transmission
        distance_factor = self.kernel.exponential_kernel(distance, lambda_param=3.0)
        
        # Effective contact rate between cells
        beta_between = self.disease_params.beta_base * 0.3 * distance_factor  # 30% of within-cell rate (higher mobility)
        
        # Apply intervention effects
        beta_between *= intervention_effects['contact_reduction']      # Contact reduction
        beta_between *= intervention_effects['transmission_reduction'] # Transmission reduction
        
        # FOI from cell1 to cell2
        prevalence_cell1 = cell1.I / (cell1.S + cell1.E + cell1.I + cell1.R)
        lambda_param = beta_between * cell2.S * prevalence_cell1
        
        new_infections = self.rng.poisson(lambda_param)
        new_infections = min(new_infections, cell2.S)
        
        cell2.S -= new_infections
        cell2.E += new_infections
    
    def disease_progression_cell(self, cell: GridCell):
        """
        Disease progression within a cell
        E → I → R/D transitions
        """
        # E → I transitions
        if cell.E > 0:
            # Sample transition times
            transition_times = self.distributions.sample_incubation_period(cell.E)
            
            # Those who completed incubation (simplified: use mean)
            mean_incubation = self.disease_params.incubation_mean
            p_transition = self.config.dt / mean_incubation
            
            n_transitions = self.rng.binomial(cell.E, p_transition)
            
            cell.E -= n_transitions
            cell.I += n_transitions
        
        # I → R/D transitions
        if cell.I > 0:
            mean_infectious = self.disease_params.infectious_shape * self.disease_params.infectious_scale
            p_transition = self.config.dt / mean_infectious
            
            n_transitions = self.rng.binomial(cell.I, p_transition)
            
            # Determine deaths
            # Simplified: use average IFR (would need age distribution for precision)
            avg_ifr = 0.01  # 1% IFR
            n_deaths = self.rng.binomial(n_transitions, avg_ifr)
            n_recoveries = n_transitions - n_deaths
            
            cell.I -= n_transitions
            cell.R += n_recoveries
            cell.D += n_deaths
    
    def step(self):
        """Execute one time step of spatial simulation"""
        
        # 0. Apply realistic interventions (if enabled)
        intervention_effects = {'contact_reduction': 1.0, 'mobility_reduction': 1.0, 'transmission_reduction': 1.0}
        if self.interventions:
            intervention_effects = self.interventions.apply_realistic_interventions(
                self.current_day, 
                self.grid.cells
            )
        
        # 1. Within-cell transmission (with intervention effects)
        for cell in self.grid.cells.values():
            self.transmission_within_cell(cell, intervention_effects)
        
        # 2. Between-cell transmission (spatial coupling, with intervention effects)
        for (x1, y1), cell1 in self.grid.cells.items():
            # Only check nearby neighbors for efficiency
            neighbors = self.grid.get_neighbors(x1, y1, radius=2)
            
            for cell2 in neighbors:
                distance = self.kernel.euclidean_distance(x1, y1, cell2.x, cell2.y)
                self.transmission_between_cells(cell1, cell2, distance, intervention_effects)
        
        # 3. Disease progression
        for cell in self.grid.cells.values():
            self.disease_progression_cell(cell)
        
        # 4. Mobility (infected people move, with intervention effects)
        effective_travel_rate = self.config.infected_travel_rate * intervention_effects['mobility_reduction']
        self.mobility.simulate_daily_mobility(infected_travel_rate=effective_travel_rate)
        
        # 5. Record state
        self._record_state()
        
        # 6. Increment time
        self.current_day += 1
    
    def _record_state(self):
        """Record current state"""
        # Aggregate totals
        totals = self.grid.get_total_counts()
        
        record = {
            'day': self.current_day,
            'S': totals['S'],
            'E': totals['E'],
            'I': totals['I'],
            'R': totals['R'],
            'D': totals['D'],
            'n_infected_cells': sum(1 for cell in self.grid.cells.values() if cell.I > 0)
        }
        
        self.history.append(record)
        
        # Record cell-level data every day for proper visualization
        if True:  # Record every day
            cell_snapshot = []
            for (x, y), cell in self.grid.cells.items():
                cell_snapshot.append({
                    'day': self.current_day,
                    'x': x,
                    'y': y,
                    'S': cell.S,
                    'E': cell.E,
                    'I': cell.I,
                    'R': cell.R,
                    'D': cell.D
                })
            self.cell_history.extend(cell_snapshot)
    
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """Run full spatial simulation"""
        
        # Initialize
        self.initialize_epidemic()
        self._record_state()
        
        if verbose:
            print(f"\nSpatial SEIR Simulation")
            print("=" * 60)
            print(f"Grid: {self.grid.rows} × {self.grid.cols} cells")
            print(f"Population: {self.grid.total_population:,}")
            print(f"Duration: {self.config.total_days} days")
            print(f"Estimated R0: {self.disease_params.R0_estimate:.2f}\n")
        
        # Main loop
        for day in range(self.config.total_days):
            self.step()
            
            if verbose and day % 30 == 0:
                rec = self.history[-1]
                print(f"Day {day:3d}: I={rec['I']:7d}, R={rec['R']:8d}, D={rec['D']:5d}, "
                      f"Cells infected: {rec['n_infected_cells']:3d}")
        
        if verbose:
            final = self.history[-1]
            print(f"\nFinal state:")
            print(f"  Total deaths: {final['D']:,}")
            print(f"  Attack rate: {100*(final['R'] + final['D'])/self.grid.total_population:.1f}%")
            print(f"  Peak infections: {max(h['I'] for h in self.history):,}")
        
        return pd.DataFrame(self.history)
    
    def get_results(self) -> pd.DataFrame:
        """Get aggregate results"""
        return pd.DataFrame(self.history)
    
    def get_cell_results(self) -> pd.DataFrame:
        """Get cell-level results"""
        return pd.DataFrame(self.cell_history)


def plot_spatial_results(simulator: SpatialSEIRSimulator, save_prefix: str = "spatial"):
    """Create comprehensive visualization of spatial results"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    results = simulator.get_results()
    
    # Figure 1: Aggregate time series
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # SEIR curves
    ax = axes[0, 0]
    ax.plot(results['day'], results['S'], label='S', linewidth=2)
    ax.plot(results['day'], results['E'], label='E', linewidth=2)
    ax.plot(results['day'], results['I'], label='I', linewidth=2)
    ax.plot(results['day'], results['R'], label='R', linewidth=2)
    ax.plot(results['day'], results['D'], label='D', linewidth=2, linestyle='--')
    ax.set_xlabel('Days')
    ax.set_ylabel('Count')
    ax.set_title('SEIR Dynamics (Total)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Active infections
    ax = axes[0, 1]
    ax.plot(results['day'], results['I'], color='red', linewidth=2)
    ax.set_xlabel('Days')
    ax.set_ylabel('Active Infections')
    ax.set_title('Active Infections Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Spatial spread
    ax = axes[1, 0]
    ax.plot(results['day'], results['n_infected_cells'], color='orange', linewidth=2)
    ax.set_xlabel('Days')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Geographic Spread', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cumulative deaths
    ax = axes[1, 1]
    ax.plot(results['day'], results['D'], color='black', linewidth=2)
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Deaths')
    ax.set_title('Mortality', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_timeseries.png', dpi=150, bbox_inches='tight')
    
    # Figure 2: Spatial snapshots at different time points
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    time_points = [0, 60, 120, 180, 240, 300]
    
    # Get cell history data
    cell_df = simulator.get_cell_results()
    
    # Find max infections for consistent color scaling
    max_infections = 0
    if len(cell_df) > 0:
        max_infections = cell_df['I'].max()
    
    for idx, day in enumerate(time_points):
        ax = axes[idx // 3, idx % 3]
        
        # Find closest recorded day in cell history
        if len(cell_df) > 0:
            available_days = cell_df['day'].unique()
            closest_day = min(available_days, key=lambda x: abs(x - day))
            day_data = cell_df[cell_df['day'] == closest_day]
            
            # Reconstruct infection map for that specific day
            infection_map = np.zeros((simulator.grid.rows, simulator.grid.cols))
            for _, row in day_data.iterrows():
                infection_map[int(row['x']), int(row['y'])] = row['I']
        else:
            # Fallback: empty map
            infection_map = np.zeros((simulator.grid.rows, simulator.grid.cols))
        
        # Use consistent color scale across all time points
        vmax = max(max_infections, 1) if max_infections > 0 else 1
        im = ax.imshow(infection_map, cmap='Reds', interpolation='nearest', 
                      vmin=0, vmax=vmax)
        ax.set_title(f'Day {day}', fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Spatial Spread of Infections', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_spatial_spread.png', dpi=150, bbox_inches='tight')
    
    print(f"Plots saved as '{save_prefix}_timeseries.png' and '{save_prefix}_spatial_spread.png'")


if __name__ == "__main__":
    print("Spatial SEIR Simulator Test")
    print("=" * 70)
    
    # Configure simulation
    config = SpatialSimulationConfig(
        total_days=365,
        grid_rows=20,
        grid_cols=20,
        total_population=500_000,
        population_distribution='clustered',
        initial_infections=5,
        seed=42
    )
    
    # Run simulation
    simulator = SpatialSEIRSimulator(config)
    results = simulator.run(verbose=True)
    
    # Visualize
    plot_spatial_results(simulator, save_prefix="test_spatial")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print(f"  Peak infections: {results['I'].max():,} on day {results['I'].idxmax()}")
    print(f"  Total deaths: {results['D'].iloc[-1]:,}")
    print(f"  Attack rate: {100 * (results['R'].iloc[-1] + results['D'].iloc[-1]) / config.total_population:.1f}%")
    print(f"  Maximum geographic spread: {results['n_infected_cells'].max()} cells")
