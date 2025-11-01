"""
Distance Kernels and Gravity Models
===================================
Implements distance-dependent transmission and mobility patterns
"""

import numpy as np
from typing import Tuple, Callable
from scipy.stats import truncnorm


class DistanceKernel:
    """
    Distance-based transmission and mobility kernels
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
    
    @staticmethod
    def euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    @staticmethod
    def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate Manhattan (city block) distance"""
        return abs(x1 - x2) + abs(y1 - y2)
    
    def exponential_kernel(self, distance: float, lambda_param: float = 5.0) -> float:
        """
        Exponential decay kernel
        P ∝ exp(-distance / λ)
        
        Args:
            distance: Distance between locations
            lambda_param: Characteristic distance (higher = longer range)
        
        Returns:
            Relative probability/weight
        """
        return np.exp(-distance / lambda_param)
    
    def power_law_kernel(self, distance: float, exponent: float = 2.0, offset: float = 1.0) -> float:
        """
        Power law decay kernel
        P ∝ 1 / (distance + offset)^exponent
        
        Args:
            distance: Distance between locations
            exponent: Decay exponent (typical: 1-3)
            offset: Offset to avoid division by zero
        
        Returns:
            Relative probability/weight
        """
        return 1.0 / ((distance + offset) ** exponent)
    
    def gaussian_kernel(self, distance: float, sigma: float = 3.0) -> float:
        """
        Gaussian kernel
        P ∝ exp(-distance² / (2σ²))
        
        Args:
            distance: Distance between locations
            sigma: Width parameter
        
        Returns:
            Relative probability/weight
        """
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))


class GravityModel:
    """
    Gravity model for human mobility
    Flow between locations i and j:
    F_ij ∝ (P_i × P_j) / distance^β × exp(-distance/λ)
    
    Combines:
    - Population attraction (larger cities attract more people)
    - Distance deterrence (farther = less travel)
    """
    
    def __init__(self,
                 distance_exponent: float = 2.0,
                 distance_decay: float = 5.0,
                 min_distance: float = 1.0,
                 seed: int = None):
        """
        Initialize gravity model
        
        Args:
            distance_exponent: Power law exponent for distance
            distance_decay: Exponential decay parameter
            min_distance: Minimum distance to avoid division by zero
            seed: Random seed
        """
        self.distance_exponent = distance_exponent
        self.distance_decay = distance_decay
        self.min_distance = min_distance
        self.rng = np.random.default_rng(seed)
        self.kernel = DistanceKernel(seed)
    
    def calculate_flow_strength(self,
                                pop_i: int,
                                pop_j: int,
                                distance: float) -> float:
        """
        Calculate relative flow strength between two locations
        
        Args:
            pop_i: Population of origin
            pop_j: Population of destination
            distance: Distance between locations
        
        Returns:
            Relative flow strength (unnormalized)
        """
        # Avoid division by zero
        distance = max(distance, self.min_distance)
        
        # Gravity term: product of populations
        gravity = pop_i * pop_j
        
        # Distance decay: power law × exponential
        power_decay = 1.0 / (distance ** self.distance_exponent)
        exp_decay = np.exp(-distance / self.distance_decay)
        
        return gravity * power_decay * exp_decay
    
    def calculate_travel_probability(self,
                                    origin_pop: int,
                                    dest_pops: np.ndarray,
                                    distances: np.ndarray) -> np.ndarray:
        """
        Calculate probability distribution over destinations
        
        Args:
            origin_pop: Population of origin
            dest_pops: Array of destination populations
            distances: Array of distances to each destination
        
        Returns:
            Probability array (sums to 1)
        """
        flows = np.array([
            self.calculate_flow_strength(origin_pop, dest_pop, dist)
            for dest_pop, dist in zip(dest_pops, distances)
        ])
        
        # Normalize to probabilities
        total_flow = flows.sum()
        if total_flow > 0:
            return flows / total_flow
        else:
            # Uniform if all flows are zero
            return np.ones_like(flows) / len(flows)
    
    def sample_destination(self,
                          origin_pop: int,
                          dest_pops: np.ndarray,
                          distances: np.ndarray) -> int:
        """
        Sample a destination index based on gravity model
        
        Returns:
            Index of chosen destination
        """
        probs = self.calculate_travel_probability(origin_pop, dest_pops, distances)
        return self.rng.choice(len(probs), p=probs)


class MobilityPatterns:
    """
    Different mobility pattern types
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
    
    def levy_flight_step(self,
                        current_x: int,
                        current_y: int,
                        grid_rows: int,
                        grid_cols: int,
                        alpha: float = 1.5) -> Tuple[int, int]:
        """
        Sample step from Lévy flight distribution
        Heavy-tailed: mostly short steps, occasional long jumps
        
        Args:
            current_x, current_y: Current position
            grid_rows, grid_cols: Grid boundaries
            alpha: Lévy exponent (1 < α ≤ 2)
        
        Returns:
            New position (x, y)
        """
        # Sample step size from power law
        # P(r) ∝ r^(-α)
        u = self.rng.uniform(0, 1)
        r = u ** (-1.0 / (alpha - 1))
        r = min(r, max(grid_rows, grid_cols))  # Cap at grid size
        
        # Random direction
        theta = self.rng.uniform(0, 2 * np.pi)
        
        dx = int(r * np.cos(theta))
        dy = int(r * np.sin(theta))
        
        # New position with boundary conditions
        new_x = np.clip(current_x + dx, 0, grid_rows - 1)
        new_y = np.clip(current_y + dy, 0, grid_cols - 1)
        
        return new_x, new_y
    
    def local_diffusion_step(self,
                            current_x: int,
                            current_y: int,
                            grid_rows: int,
                            grid_cols: int,
                            sigma: float = 1.0) -> Tuple[int, int]:
        """
        Sample step from local diffusion (Gaussian)
        Most movement is nearby
        
        Args:
            current_x, current_y: Current position
            sigma: Standard deviation of Gaussian step
        
        Returns:
            New position (x, y)
        """
        dx = int(self.rng.normal(0, sigma))
        dy = int(self.rng.normal(0, sigma))
        
        new_x = np.clip(current_x + dx, 0, grid_rows - 1)
        new_y = np.clip(current_y + dy, 0, grid_cols - 1)
        
        return new_x, new_y
    
    def commuting_pattern(self,
                         home_x: int,
                         home_y: int,
                         work_x: int,
                         work_y: int,
                         time_of_day: float) -> Tuple[int, int]:
        """
        Commuting pattern: home during night, work during day
        
        Args:
            home_x, home_y: Home location
            work_x, work_y: Work location
            time_of_day: Hour (0-24)
        
        Returns:
            Current location
        """
        # Work hours: 8am - 6pm
        if 8 <= time_of_day < 18:
            return work_x, work_y
        else:
            return home_x, home_y


def plot_distance_kernels():
    """Visualize different distance kernels"""
    import matplotlib.pyplot as plt
    
    kernel = DistanceKernel()
    distances = np.linspace(0, 20, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Exponential
    ax = axes[0]
    for lambda_param in [2, 5, 10]:
        probs = [kernel.exponential_kernel(d, lambda_param) for d in distances]
        ax.plot(distances, probs, label=f'λ={lambda_param}', linewidth=2)
    ax.set_title('Exponential Kernel', fontweight='bold')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Relative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Power law
    ax = axes[1]
    for exp in [1, 2, 3]:
        probs = [kernel.power_law_kernel(d, exp) for d in distances]
        ax.plot(distances, probs, label=f'α={exp}', linewidth=2)
    ax.set_title('Power Law Kernel', fontweight='bold')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Relative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Gaussian
    ax = axes[2]
    for sigma in [2, 4, 6]:
        probs = [kernel.gaussian_kernel(d, sigma) for d in distances]
        ax.plot(distances, probs, label=f'σ={sigma}', linewidth=2)
    ax.set_title('Gaussian Kernel', fontweight='bold')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Relative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Distance Kernel Test")
    print("=" * 60)
    
    # Test gravity model
    gravity = GravityModel(seed=42)
    
    # Example: flow from city A to cities B, C, D
    origin_pop = 100000
    dest_pops = np.array([50000, 200000, 30000])
    distances = np.array([5, 15, 3])
    
    probs = gravity.calculate_travel_probability(origin_pop, dest_pops, distances)
    
    print("Gravity Model Example:")
    print(f"Origin population: {origin_pop:,}")
    print(f"\nDestinations:")
    for i, (pop, dist, prob) in enumerate(zip(dest_pops, distances, probs)):
        print(f"  City {i+1}: pop={pop:,}, distance={dist}, P(travel)={prob:.3f}")
    
    # Test sampling
    print(f"\nSampling 1000 destinations:")
    samples = [gravity.sample_destination(origin_pop, dest_pops, distances) for _ in range(1000)]
    unique, counts = np.unique(samples, return_counts=True)
    for dest, count in zip(unique, counts):
        print(f"  City {dest+1}: {count} times ({100*count/1000:.1f}%)")
    
    # Visualize kernels
    fig = plot_distance_kernels()
    plt.savefig('distance_kernels.png', dpi=150, bbox_inches='tight')
    print("\nKernel plot saved as 'distance_kernels.png'")
