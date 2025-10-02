"""
Entity runtime representation.

Entities are spawned from species definitions and exist in the simulation.
Each entity has a unique instance_id, position, velocity, and size variation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Entity:
    """
    Runtime entity in simulation.

    Attributes:
        instance_id: Unique identifier (format: "{species_id}-{biome_id}-{index:04d}")
        species_id: Species definition ID (e.g., "sp-001-drifter")
        biome_id: Biome this entity belongs to
        position: 3D position [x, y, z] in meters (y = depth, negative)
        velocity: 3D velocity [vx, vy, vz] in m/s
        size_factor: Size variation (1.0 = nominal, 0.8-1.2 typical range)
        tags: List of tags (e.g., ['mobile', 'predator', 'ship'])
        active_behavior_id: Currently active behavior ID (e.g., "flee-predator")
        emission_multipliers: Per-channel emission multipliers from active behavior
        knowledge_tokens: Knowledge tokens (Phase 6+, placeholder for now)
    """
    instance_id: str
    species_id: str
    biome_id: str
    position: np.ndarray  # [x, y, z] float64
    velocity: np.ndarray  # [vx, vy, vz] float64
    size_factor: float
    tags: list  # Species tags + special tags (e.g., 'ship')
    active_behavior_id: str = None  # Current behavior
    emission_multipliers: dict = None  # {channel: multiplier}
    base_emissions: dict = None  # {channel: {field: value}}, baked at spawn from Species
    knowledge_tokens: dict = None  # {token_kind: token_data}, Phase 6+

    def __post_init__(self):
        """Ensure position and velocity are float64 arrays, initialize defaults"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        else:
            self.position = self.position.astype(np.float64, copy=False)

        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)
        else:
            self.velocity = self.velocity.astype(np.float64, copy=False)

        # Initialize emission multipliers (1.0 = baseline)
        if self.emission_multipliers is None:
            self.emission_multipliers = {
                'acoustic': 1.0,
                'thermal': 1.0,
                'chemical': 1.0,
                'magnetic': 1.0,
                'bioluminescent': 1.0
            }

        # Initialize base emissions (baked at spawn, empty if not provided)
        if self.base_emissions is None:
            self.base_emissions = {}

        # Initialize knowledge tokens
        if self.knowledge_tokens is None:
            self.knowledge_tokens = {}

    def update_position(self, dt: float):
        """
        Update position using current velocity.

        Args:
            dt: Time step in seconds
        """
        self.position += self.velocity * dt

    def get_emission_multipliers(self) -> dict:
        """
        Get current emission multipliers for sensor queries.

        Returns:
            Dict of {channel: multiplier} (e.g., {'acoustic': 0.3, 'bioluminescent': 1.5})
        """
        return self.emission_multipliers.copy()

    def to_dict(self) -> dict:
        """
        Serialize entity to JSON-compatible dict.

        Returns:
            Dict with all entity fields
        """
        return {
            'instance_id': self.instance_id,
            'species_id': self.species_id,
            'biome_id': self.biome_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'size_factor': self.size_factor,
            'tags': self.tags,
            'active_behavior_id': self.active_behavior_id,
            'emission_multipliers': self.emission_multipliers,
            'base_emissions': self.base_emissions,
            'knowledge_tokens': self.knowledge_tokens
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Entity':
        """
        Deserialize entity from dict.

        Args:
            data: Dict with entity fields

        Returns:
            Entity instance
        """
        return cls(
            instance_id=data['instance_id'],
            species_id=data['species_id'],
            biome_id=data['biome_id'],
            position=np.array(data['position'], dtype=np.float64),
            velocity=np.array(data['velocity'], dtype=np.float64),
            size_factor=data['size_factor'],
            tags=data.get('tags', []),
            active_behavior_id=data.get('active_behavior_id'),
            emission_multipliers=data.get('emission_multipliers'),
            base_emissions=data.get('base_emissions'),
            knowledge_tokens=data.get('knowledge_tokens')
        )
