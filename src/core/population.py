"""
Population Management
====================
Creates and manages age-structured population with realistic demographics
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from enum import IntEnum

class DiseaseState(IntEnum):
    """Enumeration of disease states"""
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTED = 2
    RECOVERED = 3
    DEAD = 4


@dataclass
class Person:
    """Individual agent in the simulation"""
    id: int
    age: int
    household_id: int
    occupation: str
    has_comorbidity: bool
    
    # Disease state
    state: DiseaseState = DiseaseState.SUSCEPTIBLE
    
    # Disease progression timers
    incubation_time: float = 0.0  # Time until E→I
    infectious_time: float = 0.0  # Time until I→R/D
    time_in_state: float = 0.0    # Days in current state
    
    # Disease characteristics (assigned upon infection)
    severity: Optional[str] = None
    will_die: bool = False
    
    # Location (for spatial model - Phase 2)
    grid_x: int = 0
    grid_y: int = 0


class Population:
    """
    Manages the population structure with realistic demographics
    """
    
    def __init__(self, 
                 size: int,
                 age_distribution: Optional[Dict[str, float]] = None,
                 seed: int = None):
        """
        Initialize population
        
        Args:
            size: Total population size
            age_distribution: Dict of age group → proportion
            seed: Random seed for reproducibility
        """
        self.size = size
        self.rng = np.random.default_rng(seed)
        
        # Default age distribution (approximate US demographics)
        if age_distribution is None:
            age_distribution = {
                '0-9': 0.12,
                '10-19': 0.13,
                '20-29': 0.14,
                '30-39': 0.13,
                '40-49': 0.13,
                '50-59': 0.13,
                '60-69': 0.12,
                '70-79': 0.07,
                '80+': 0.03
            }
        
        self.age_distribution = age_distribution
        self.people = self._initialize_people()
        
        # State counts
        self.state_counts = {
            DiseaseState.SUSCEPTIBLE: size,
            DiseaseState.EXPOSED: 0,
            DiseaseState.INFECTED: 0,
            DiseaseState.RECOVERED: 0,
            DiseaseState.DEAD: 0
        }
    
    def _sample_age_from_group(self, age_group: str) -> int:
        """Sample an age uniformly within an age group"""
        if age_group == '0-9':
            return self.rng.integers(0, 10)
        elif age_group == '10-19':
            return self.rng.integers(10, 20)
        elif age_group == '20-29':
            return self.rng.integers(20, 30)
        elif age_group == '30-39':
            return self.rng.integers(30, 40)
        elif age_group == '40-49':
            return self.rng.integers(40, 50)
        elif age_group == '50-59':
            return self.rng.integers(50, 60)
        elif age_group == '60-69':
            return self.rng.integers(60, 70)
        elif age_group == '70-79':
            return self.rng.integers(70, 80)
        else:  # '80+'
            return self.rng.integers(80, 100)
    
    def _assign_occupation(self, age: int) -> str:
        """Assign occupation based on age"""
        if age < 18:
            return 'student'
        elif age < 65:
            # Working age
            occupations = ['essential', 'remote', 'healthcare']
            probs = [0.3, 0.6, 0.1]
            return self.rng.choice(occupations, p=probs)
        else:
            return 'retired'
    
    def _has_comorbidity(self, age: int) -> bool:
        """Determine if person has comorbidity (age-dependent)"""
        # Probability increases with age
        if age < 40:
            prob = 0.05
        elif age < 60:
            prob = 0.15
        elif age < 70:
            prob = 0.30
        else:
            prob = 0.50
        
        return self.rng.random() < prob
    
    def _initialize_people(self) -> list[Person]:
        """Create all individuals with attributes"""
        people = []
        
        # Determine number in each age group
        age_groups = list(self.age_distribution.keys())
        proportions = list(self.age_distribution.values())
        
        age_group_counts = self.rng.multinomial(self.size, proportions)
        
        person_id = 0
        household_id = 0
        household_size_target = 0
        household_members = 0
        
        for age_group, count in zip(age_groups, age_group_counts):
            for _ in range(count):
                # Assign household (simple clustering)
                if household_members >= household_size_target:
                    household_id += 1
                    # Sample household size: 1-5 people
                    household_size_target = self.rng.integers(1, 6)
                    household_members = 0
                
                age = self._sample_age_from_group(age_group)
                
                person = Person(
                    id=person_id,
                    age=age,
                    household_id=household_id,
                    occupation=self._assign_occupation(age),
                    has_comorbidity=self._has_comorbidity(age)
                )
                
                people.append(person)
                person_id += 1
                household_members += 1
        
        return people
    
    def get_state_counts(self) -> Dict[DiseaseState, int]:
        """Count people in each disease state"""
        counts = {state: 0 for state in DiseaseState}
        for person in self.people:
            counts[person.state] += 1
        return counts
    
    def get_susceptible(self) -> list[Person]:
        """Get all susceptible individuals"""
        return [p for p in self.people if p.state == DiseaseState.SUSCEPTIBLE]
    
    def get_exposed(self) -> list[Person]:
        """Get all exposed individuals"""
        return [p for p in self.people if p.state == DiseaseState.EXPOSED]
    
    def get_infected(self) -> list[Person]:
        """Get all infected individuals"""
        return [p for p in self.people if p.state == DiseaseState.INFECTED]
    
    def get_age_distribution(self) -> Dict[str, int]:
        """Get count of people in each age group"""
        age_groups = {
            '0-9': 0, '10-19': 0, '20-29': 0, '30-39': 0, '40-49': 0,
            '50-59': 0, '60-69': 0, '70-79': 0, '80+': 0
        }
        
        for person in self.people:
            if person.age < 10:
                age_groups['0-9'] += 1
            elif person.age < 20:
                age_groups['10-19'] += 1
            elif person.age < 30:
                age_groups['20-29'] += 1
            elif person.age < 40:
                age_groups['30-39'] += 1
            elif person.age < 50:
                age_groups['40-49'] += 1
            elif person.age < 60:
                age_groups['50-59'] += 1
            elif person.age < 70:
                age_groups['60-69'] += 1
            elif person.age < 80:
                age_groups['70-79'] += 1
            else:
                age_groups['80+'] += 1
        
        return age_groups
    
    def summary(self) -> str:
        """Return population summary statistics"""
        ages = [p.age for p in self.people]
        
        summary = f"Population Summary\n"
        summary += f"=" * 50 + "\n"
        summary += f"Total size: {self.size}\n"
        summary += f"Age range: {min(ages)} - {max(ages)}\n"
        summary += f"Mean age: {np.mean(ages):.1f}\n"
        summary += f"Median age: {np.median(ages):.1f}\n\n"
        
        summary += f"Age distribution:\n"
        for group, count in self.get_age_distribution().items():
            pct = 100 * count / self.size
            summary += f"  {group:8s}: {count:6d} ({pct:5.1f}%)\n"
        
        # Occupation distribution
        occupations = {}
        for person in self.people:
            occupations[person.occupation] = occupations.get(person.occupation, 0) + 1
        
        summary += f"\nOccupation distribution:\n"
        for occ, count in sorted(occupations.items()):
            pct = 100 * count / self.size
            summary += f"  {occ:12s}: {count:6d} ({pct:5.1f}%)\n"
        
        # Comorbidity
        with_comorbidity = sum(1 for p in self.people if p.has_comorbidity)
        summary += f"\nWith comorbidities: {with_comorbidity} ({100*with_comorbidity/self.size:.1f}%)\n"
        
        return summary


if __name__ == "__main__":
    # Test population creation
    pop = Population(size=10000, seed=42)
    print(pop.summary())
    
    # Test state tracking
    print("\n" + "=" * 50)
    print("Initial disease state counts:")
    for state, count in pop.get_state_counts().items():
        print(f"  {state.name}: {count}")
