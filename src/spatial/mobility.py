"""
Mobility Model
==============
Manages movement of people between grid cells
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .grid import SpatialGrid, GridCell
from .distance_kernel import GravityModel, DistanceKernel


@dataclass
class MovementConfig:
    """Configuration for mobility patterns"""
    # Daily travel probabilities
    stay_local: float = 0.90  # Probability of staying in home cell
    neighbor_travel: float = 0.08  # Travel to adjacent cell
    long_distance: float = 0.02  # Travel to distant cell
    
    # Gravity model parameters
    distance_exponent: float = 2.0
    distance_decay: float = 5.0
    
    # Intervention modifier (lockdown reduces mobility)
    mobility_reduction: float = 1.0  # 1.0 = normal, 0.2 = strict lockdown


class MobilitySimulator:
    """
    Simulates human mobility patterns on spatial grid
    """
    
    def __init__(self,
                 grid: SpatialGrid,
                 config: MovementConfig,
                 seed: int = None):
        """
        Initialize mobility simulator
        
        Args:
            grid: SpatialGrid object
            config: MovementConfig object
            seed: Random seed
        """
        self.grid = grid
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Gravity model for long-distance travel
        self.gravity = GravityModel(
            distance_exponent=config.distance_exponent,
            distance_decay=config.distance_decay,
            seed=seed
        )
        
        self.kernel = DistanceKernel(seed=seed)
        
        # Cache: precompute travel probabilities for efficiency
        self._travel_cache = {}
        self._build_travel_cache()
    
    def _build_travel_cache(self):
        """
        Precompute travel probabilities from each cell to all others
        Speeds up simulation
        """
        print("Building travel probability cache...")
        
        for (x1, y1), cell1 in self.grid.cells.items():
            destinations = []
            dest_pops = []
            distances = []
            
            for (x2, y2), cell2 in self.grid.cells.items():
                if (x1, y1) != (x2, y2):
                    destinations.append((x2, y2))
                    dest_pops.append(cell2.population)
                    distances.append(self.kernel.euclidean_distance(x1, y1, x2, y2))
            
            # Calculate probabilities using gravity model
            probs = self.gravity.calculate_travel_probability(
                cell1.population,
                np.array(dest_pops),
                np.array(distances)
            )
            
            self._travel_cache[(x1, y1)] = {
                'destinations': destinations,
                'probabilities': probs
            }
    
    def sample_destination(self, origin_x: int, origin_y: int) -> Tuple[int, int]:
        """
        Sample travel destination from origin cell
        Uses cached probabilities for efficiency
        
        Returns:
            Destination cell coordinates (x, y)
        """
        cache_data = self._travel_cache[(origin_x, origin_y)]
        
        idx = self.rng.choice(
            len(cache_data['destinations']),
            p=cache_data['probabilities']
        )
        
        return cache_data['destinations'][idx]
    
    def determine_travel_type(self) -> str:
        """
        Determine if person travels and how far
        Returns: 'local', 'neighbor', 'long_distance'
        """
        # Apply mobility reduction from interventions
        effective_travel_prob = (1 - self.config.stay_local) * self.config.mobility_reduction
        
        r = self.rng.random()
        
        if r > effective_travel_prob:
            return 'local'  # Stay in current cell
        
        # Among travelers, choose travel distance
        travel_dist = self.rng.random()
        threshold = self.config.neighbor_travel / (self.config.neighbor_travel + self.config.long_distance)
        
        if travel_dist < threshold:
            return 'neighbor'
        else:
            return 'long_distance'
    
    def get_neighbor_destination(self, x: int, y: int) -> Tuple[int, int]:
        """
        Choose a random neighboring cell
        """
        neighbors = self.grid.get_neighbors(x, y, radius=1)
        
        if not neighbors:
            return x, y  # Stay if no neighbors (shouldn't happen)
        
        # Weighted by population (more likely to travel to populated areas)
        pops = np.array([n.population for n in neighbors])
        probs = pops / pops.sum() if pops.sum() > 0 else np.ones(len(pops)) / len(pops)
        
        chosen = self.rng.choice(neighbors, p=probs)
        return chosen.x, chosen.y
    
    def move_infected_people(self, n_movers_per_cell: Dict[Tuple[int, int], int]) -> Dict[str, int]:
        """
        Move infected people between cells
        
        Args:
            n_movers_per_cell: Dict mapping (x,y) to number of infected people who travel
        
        Returns:
            Dict with movement statistics
        """
        movements = {
            'local': 0,
            'neighbor': 0,
            'long_distance': 0
        }
        
        for (origin_x, origin_y), n_movers in n_movers_per_cell.items():
            origin_cell = self.grid.get_cell(origin_x, origin_y)
            
            if n_movers == 0 or origin_cell.I == 0:
                continue
            
            # Don't move more infected than exist
            n_movers = min(n_movers, origin_cell.I)
            
            for _ in range(n_movers):
                travel_type = self.determine_travel_type()
                movements[travel_type] += 1
                
                if travel_type == 'local':
                    # Stay in same cell
                    continue
                
                elif travel_type == 'neighbor':
                    dest_x, dest_y = self.get_neighbor_destination(origin_x, origin_y)
                
                else:  # long_distance
                    dest_x, dest_y = self.sample_destination(origin_x, origin_y)
                
                # Move one infected person
                if dest_x != origin_x or dest_y != origin_y:
                    origin_cell.I -= 1
                    dest_cell = self.grid.get_cell(dest_x, dest_y)
                    dest_cell.I += 1
        
        return movements
    
    def simulate_daily_mobility(self, infected_travel_rate: float = 0.1) -> Dict[str, int]:
        """
        Simulate one day of mobility
        
        Args:
            infected_travel_rate: Fraction of infected who travel (reduced due to illness)
        
        Returns:
            Movement statistics
        """
        # Determine how many infected people travel from each cell
        n_movers = {}
        
        for (x, y), cell in self.grid.cells.items():
            if cell.I > 0:
                # Number who travel = Binomial(I, travel_rate)
                n_travel = self.rng.binomial(cell.I, infected_travel_rate)
                n_movers[(x, y)] = n_travel
        
        # Execute movements
        return self.move_infected_people(n_movers)
    
    def get_travel_matrix(self) -> np.ndarray:
        """
        Get travel flow matrix between all cell pairs
        Useful for visualization
        
        Returns:
            Matrix of shape (n_cells, n_cells) with flow strengths
        """
        n_cells = len(self.grid.cells)
        cells_list = list(self.grid.cells.keys())
        
        flow_matrix = np.zeros((n_cells, n_cells))
        
        for i, (x1, y1) in enumerate(cells_list):
            cell1 = self.grid.get_cell(x1, y1)
            
            for j, (x2, y2) in enumerate(cells_list):
                if i != j:
                    cell2 = self.grid.get_cell(x2, y2)
                    distance = self.kernel.euclidean_distance(x1, y1, x2, y2)
                    
                    flow = self.gravity.calculate_flow_strength(
                        cell1.population,
                        cell2.population,
                        distance
                    )
                    
                    flow_matrix[i, j] = flow
        
        return flow_matrix


def visualize_mobility_network(grid: SpatialGrid, mobility: MobilitySimulator):
    """
    Visualize mobility connections between cells
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Spatial layout with connections
    G = nx.Graph()
    
    # Add nodes
    positions = {}
    for (x, y), cell in grid.cells.items():
        G.add_node((x, y), population=cell.population)
        positions[(x, y)] = (y, -x)  # Flip x for proper orientation
    
    # Add edges (show only strong connections for clarity)
    threshold = 0.01  # Only show flows above this threshold
    
    for (x1, y1) in grid.cells.keys():
        cache = mobility._travel_cache[(x1, y1)]
        
        for (x2, y2), prob in zip(cache['destinations'], cache['probabilities']):
            if prob > threshold:
                G.add_edge((x1, y1), (x2, y2), weight=prob)
    
    # Node sizes proportional to population
    node_sizes = [grid.get_cell(x, y).population / 100 for (x, y) in G.nodes()]
    
    # Edge widths proportional to flow probability
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, positions, node_size=node_sizes, 
                          node_color='skyblue', alpha=0.7, ax=ax1)
    nx.draw_networkx_edges(G, positions, width=edge_widths, 
                          alpha=0.3, edge_color='gray', ax=ax1)
    
    ax1.set_title('Mobility Network\n(node size = population, edge width = flow)', 
                 fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Flow matrix heatmap
    flow_matrix = mobility.get_travel_matrix()
    
    im = ax2.imshow(np.log10(flow_matrix + 1), cmap='YlOrRd', interpolation='nearest')
    ax2.set_title('Travel Flow Matrix (log scale)', fontweight='bold')
    ax2.set_xlabel('Destination Cell')
    ax2.set_ylabel('Origin Cell')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('log10(Flow Strength)')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Mobility Model Test")
    print("=" * 60)
    
    # Create grid
    grid = SpatialGrid(
        rows=15,
        cols=15,
        total_population=500_000,
        population_distribution='clustered',
        seed=42
    )
    
    print(grid.summary())
    
    # Seed infections in one city
    grid.seed_infection(7, 7, n_infected=100)
    
    # Create mobility model
    config = MovementConfig()
    mobility = MobilitySimulator(grid, config, seed=42)
    
    print("\nSimulating 30 days of mobility...")
    
    total_movements = {'local': 0, 'neighbor': 0, 'long_distance': 0}
    
    for day in range(30):
        movements = mobility.simulate_daily_mobility(infected_travel_rate=0.1)
        
        for key in total_movements:
            total_movements[key] += movements[key]
        
        if day % 10 == 0:
            counts = grid.get_total_counts()
            print(f"Day {day}: {counts['I']} infected across {sum(1 for c in grid.cells.values() if c.I > 0)} cells")
    
    print(f"\nTotal movements over 30 days:")
    for travel_type, count in total_movements.items():
        pct = 100 * count / sum(total_movements.values()) if sum(total_movements.values()) > 0 else 0
        print(f"  {travel_type}: {count} ({pct:.1f}%)")
    
    # Visualize
    fig = visualize_mobility_network(grid, mobility)
    plt.savefig('mobility_network.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'mobility_network.png'")
