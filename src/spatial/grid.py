"""
Spatial Grid Structure
======================
Implements geographic structure for epidemic spread
Each grid cell represents a region with its own population and epidemic dynamics
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


@dataclass
class GridCell:
    """Represents a single geographic cell"""
    x: int
    y: int
    population: int
    density: float  # people per unit area
    
    # SEIR counts in this cell
    S: int = 0
    E: int = 0
    I: int = 0
    R: int = 0
    D: int = 0
    
    def __post_init__(self):
        self.S = self.population  # Initially all susceptible
    
    @property
    def total(self) -> int:
        """Total population (should always equal self.population minus deaths)"""
        return self.S + self.E + self.I + self.R
    
    @property
    def prevalence(self) -> float:
        """Proportion currently infected"""
        if self.total == 0:
            return 0.0
        return self.I / self.total
    
    def update_density(self):
        """Recalculate density based on current population"""
        # Assume each cell is 1x1 unit area
        self.density = self.population / 1.0


class SpatialGrid:
    """
    2D grid structure for spatial epidemic modeling
    """
    
    def __init__(self,
                 rows: int,
                 cols: int,
                 total_population: int,
                 population_distribution: str = 'lognormal',
                 seed: int = None):
        """
        Initialize spatial grid
        
        Args:
            rows: Number of rows in grid
            cols: Number of columns in grid
            total_population: Total population to distribute
            population_distribution: 'uniform', 'lognormal', or 'clustered'
            seed: Random seed
        """
        self.rows = rows
        self.cols = cols
        self.total_population = total_population
        self.rng = np.random.default_rng(seed)
        
        # Create grid
        self.cells: Dict[Tuple[int, int], GridCell] = {}
        self._initialize_grid(population_distribution)
    
    def _initialize_grid(self, distribution: str):
        """Create grid cells with population distribution"""
        n_cells = self.rows * self.cols
        
        # Generate population for each cell
        if distribution == 'uniform':
            populations = np.full(n_cells, self.total_population // n_cells)
        
        elif distribution == 'lognormal':
            # Lognormal distribution (realistic - cities vary widely in size)
            mean_pop = self.total_population / n_cells
            
            # Parameters for lognormal to achieve desired mean
            sigma = 1.0
            mu = np.log(mean_pop) - 0.5 * sigma**2
            
            populations = self.rng.lognormal(mu, sigma, n_cells)
            
            # Normalize to exact total
            populations = populations * (self.total_population / populations.sum())
            populations = populations.astype(int)
            
            # Fix rounding errors
            diff = self.total_population - populations.sum()
            populations[0] += diff
        
        elif distribution == 'clustered':
            # Create urban centers and rural areas
            populations = np.zeros(n_cells)
            
            # Create 2-3 urban centers
            n_centers = self.rng.integers(2, 4)
            urban_fraction = 0.7  # 70% of population in urban areas
            
            for _ in range(n_centers):
                # Pick random center
                center_idx = self.rng.integers(0, n_cells)
                # Allocate population
                populations[center_idx] = (urban_fraction / n_centers) * self.total_population
            
            # Distribute remaining population
            remaining = self.total_population - populations.sum()
            rural_cells = populations == 0
            populations[rural_cells] = remaining / rural_cells.sum()
            
            populations = populations.astype(int)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Create GridCell objects
        idx = 0
        for i in range(self.rows):
            for j in range(self.cols):
                pop = int(populations[idx])
                
                cell = GridCell(
                    x=i,
                    y=j,
                    population=pop,
                    density=pop / 1.0  # Assuming unit area
                )
                
                self.cells[(i, j)] = cell
                idx += 1
    
    def get_cell(self, x: int, y: int) -> GridCell:
        """Get cell at position (x, y)"""
        return self.cells.get((x, y))
    
    def get_neighbors(self, x: int, y: int, radius: int = 1) -> List[GridCell]:
        """Get neighboring cells within radius"""
        neighbors = []
        for i in range(max(0, x - radius), min(self.rows, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.cols, y + radius + 1)):
                if (i, j) != (x, y):  # Exclude self
                    cell = self.get_cell(i, j)
                    if cell is not None:
                        neighbors.append(cell)
        return neighbors
    
    def get_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance between two cells"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def seed_infection(self, x: int, y: int, n_infected: int):
        """
        Seed initial infections in a specific cell
        
        Args:
            x, y: Cell coordinates
            n_infected: Number of initial infections
        """
        cell = self.get_cell(x, y)
        if cell is None:
            raise ValueError(f"Invalid cell coordinates: ({x}, {y})")
        
        # Move people from S to I
        n_to_infect = min(n_infected, cell.S)
        cell.S -= n_to_infect
        cell.I += n_to_infect
    
    def get_total_counts(self) -> Dict[str, int]:
        """Get total SEIR counts across all cells"""
        totals = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        
        for cell in self.cells.values():
            totals['S'] += cell.S
            totals['E'] += cell.E
            totals['I'] += cell.I
            totals['R'] += cell.R
            totals['D'] += cell.D
        
        return totals
    
    def get_infection_map(self) -> np.ndarray:
        """Get 2D array of infection counts"""
        infection_map = np.zeros((self.rows, self.cols))
        
        for (x, y), cell in self.cells.items():
            infection_map[x, y] = cell.I
        
        return infection_map
    
    def get_prevalence_map(self) -> np.ndarray:
        """Get 2D array of infection prevalence (proportion)"""
        prevalence_map = np.zeros((self.rows, self.cols))
        
        for (x, y), cell in self.cells.items():
            prevalence_map[x, y] = cell.prevalence
        
        return prevalence_map
    
    def plot_population_distribution(self, ax=None):
        """Plot population heatmap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        pop_map = np.zeros((self.rows, self.cols))
        for (x, y), cell in self.cells.items():
            pop_map[x, y] = cell.population
        
        im = ax.imshow(pop_map, cmap='YlOrRd', interpolation='nearest')
        ax.set_title('Population Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid X', fontsize=12)
        ax.set_ylabel('Grid Y', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Population', fontsize=11)
        
        # Add grid
        ax.grid(False)
        
        return ax
    
    def plot_infections(self, ax=None, log_scale: bool = False):
        """Plot infection heatmap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        infection_map = self.get_infection_map()
        
        if log_scale:
            # Add 1 to avoid log(0)
            norm = LogNorm(vmin=1, vmax=infection_map.max() + 1)
            im = ax.imshow(infection_map + 1, cmap='Reds', interpolation='nearest', norm=norm)
        else:
            im = ax.imshow(infection_map, cmap='Reds', interpolation='nearest')
        
        ax.set_title('Active Infections', fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid X', fontsize=12)
        ax.set_ylabel('Grid Y', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Infected Count', fontsize=11)
        
        ax.grid(False)
        
        return ax
    
    def summary(self) -> str:
        """Return grid summary statistics"""
        pops = [cell.population for cell in self.cells.values()]
        
        summary = f"Spatial Grid Summary\n"
        summary += f"=" * 50 + "\n"
        summary += f"Dimensions: {self.rows} × {self.cols} = {len(self.cells)} cells\n"
        summary += f"Total population: {self.total_population:,}\n"
        summary += f"Population per cell:\n"
        summary += f"  Mean: {np.mean(pops):,.0f}\n"
        summary += f"  Median: {np.median(pops):,.0f}\n"
        summary += f"  Min: {np.min(pops):,.0f}\n"
        summary += f"  Max: {np.max(pops):,.0f}\n"
        summary += f"  Std: {np.std(pops):,.0f}\n"
        
        # Current epidemic state
        totals = self.get_total_counts()
        summary += f"\nCurrent epidemic state:\n"
        for state, count in totals.items():
            summary += f"  {state}: {count:,}\n"
        
        return summary


if __name__ == "__main__":
    # Test grid creation
    print("Spatial Grid Test")
    print("=" * 60)
    
    # Create grid with lognormal distribution (realistic)
    grid = SpatialGrid(
        rows=20,
        cols=20,
        total_population=1_000_000,
        population_distribution='lognormal',
        seed=42
    )
    
    print(grid.summary())
    
    # Seed some infections in the center
    center_x, center_y = 10, 10
    grid.seed_infection(center_x, center_y, n_infected=100)
    
    print(f"\nSeeded 100 infections at cell ({center_x}, {center_y})")
    print(f"Updated counts: {grid.get_total_counts()}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    grid.plot_population_distribution(ax=ax1)
    grid.plot_infections(ax=ax2)
    plt.tight_layout()
    plt.savefig('spatial_grid_test.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'spatial_grid_test.png'")
