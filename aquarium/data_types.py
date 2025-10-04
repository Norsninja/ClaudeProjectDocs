"""
Data types mirroring YAML schema structures.

These dataclasses are populated by loader.py from YAML files.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ============================================================================
# Species Definition
# ============================================================================

@dataclass
class EmissionProfile:
    """Base emission profile for a species"""
    acoustic: Optional[Dict[str, float]] = None  # {peak_hz, bandwidth_hz, amplitude}
    thermal: Optional[Dict[str, float]] = None  # {delta_celsius}
    chemical: Optional[Dict[str, float]] = None  # {compound, concentration}
    magnetic: Optional[Dict[str, float]] = None  # {delta_microtesla}
    bioluminescent: Optional[Dict[str, float]] = None  # {intensity, wavelength_nm}


@dataclass
class PhysicalProperties:
    """Physical properties of a species"""
    size_range: Dict[str, float]  # {min_meters, max_meters}
    mass_kg: float
    personal_space: float = 1.0
    avoidance_weight: float = 0.6


@dataclass
class MovementProperties:
    """Movement capabilities of a species"""
    max_speed_ms: float
    acceleration_ms2: float = 1.0
    turn_rate_deg_s: float = 90.0
    depth_range: Optional[Dict[str, float]] = None  # {shallow_limit_meters, deep_limit_meters}


@dataclass
class BehaviorCondition:
    """A single condition in a behavior rule"""
    type: str  # nearest_entity, knowledge_token, token_exists, within_depth_band
    tag: Optional[str] = None  # For nearest_entity
    max_distance: Optional[float] = None  # For nearest_entity
    token: Optional[str] = None  # For knowledge_token
    operator: Optional[str] = None  # less_than, greater_than, etc.
    value: Optional[float] = None  # For knowledge_token
    min_depth: Optional[float] = None  # For within_depth_band
    max_depth: Optional[float] = None  # For within_depth_band


@dataclass
class BehaviorAction:
    """Action to execute when behavior conditions met"""
    type: str  # flee, investigate, forage, hold, align_with_current
    speed_multiplier: float = 1.0
    emission_multipliers: Optional[Dict[str, float]] = None  # Per-channel multipliers


@dataclass
class Behavior:
    """A single behavior rule with priority"""
    id: str
    priority: int
    conditions: List[BehaviorCondition]
    action: BehaviorAction


@dataclass
class KnowledgeConfig:
    """Knowledge token configuration for species"""
    gossip_range: float = 0.0
    token_capacity: int = 10


@dataclass
class MetabolismConfig:
    """Energy and metabolism configuration for species"""
    energy_max: float  # Maximum energy capacity
    energy_drain_per_tick: float  # Passive metabolic drain
    movement_cost_factor: float = 0.1  # Energy cost per m/s velocity
    feeding_efficiency: float = 2.0  # Energy gained per unit resource consumed
    feeding_cooldown_ticks: int = 5  # Min ticks between feeding actions
    starvation_threshold: float = 0.0  # Death occurs at this energy level


@dataclass
class FeedingConfig:
    """Feeding behavior configuration for species"""
    diet: List[str] = field(default_factory=lambda: ["plankton"])  # Resource types consumed
    intake_rate_per_tick: float = 5.0  # Max units consumed per tick (prevents energy spikes)
    feeding_range_m: float = 0.0  # Feeding range (0.0 = sample at position only)


@dataclass
class Species:
    """Complete species definition"""
    species_id: str
    name: str
    tags: List[str]
    physical: PhysicalProperties
    emissions: EmissionProfile
    movement: MovementProperties
    behaviors: List[Behavior]
    knowledge: KnowledgeConfig
    metabolism: Optional[MetabolismConfig] = None
    feeding: Optional[FeedingConfig] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    description: Optional[str] = None


# ============================================================================
# Biome Definition
# ============================================================================

@dataclass
class SphereObstacle:
    """Spherical obstacle (e.g., vent)"""
    id: str
    center: List[float]  # [x, y, z]
    radius: float
    influence_radius: Optional[float] = None
    thermal_base_delta: Optional[float] = None  # Temperature delta (°C) for thermal sensors
    resource_peak: Optional[float] = None  # Peak plankton density at vent (Phase 8+: Ecosystem)
    resource_sigma: Optional[float] = None  # Gaussian falloff distance (Phase 8+: Ecosystem)


@dataclass
class CylinderObstacle:
    """Cylindrical obstacle (e.g., ridge)"""
    id: str
    start: List[float]  # [x, y, z]
    end: List[float]  # [x, y, z]
    radius: float
    influence_radius: Optional[float] = None


@dataclass
class PlaneObstacle:
    """Planar obstacle (e.g., seabed)"""
    id: str
    type: str  # "seabed"
    y_level: float
    influence_distance: float = 5.0


@dataclass
class SpawningConfig:
    """Species spawning configuration"""
    species_id: str
    count: int
    distribution: str = "uniform"  # uniform, clustered, near_obstacles


@dataclass
class Biome:
    """Biome definition with obstacles and spawning"""
    biome_id: str
    name: str
    bounds: Dict[str, Any]  # {center: [x,y,z], radius: float}
    obstacles: Dict[str, List[Any]]  # {spheres: [SphereObstacle], cylinders: [CylinderObstacle], planes: [PlaneObstacle]}
    spawning: Optional[Dict[str, List[SpawningConfig]]] = None
    seed: Optional[int] = None
    description: Optional[str] = None


# ============================================================================
# World Definition
# ============================================================================

@dataclass
class WorldParameters:
    """Physical parameters of the world"""
    gravity_ms2: float
    fluid_density_kgm3: float
    ambient_temperature_c: float
    current_speed_ms: float = 0.0
    current_direction_deg: float = 0.0
    seed: Optional[int] = None


@dataclass
class SimulationConfig:
    """Simulation global defaults"""
    tick_delta_seconds: float = 1.0
    active_region_radius: float = 2000.0
    influence_radius_factor: float = 2.5
    avoidance_weight: float = 0.6
    seabed_influence_distance: float = 5.0


@dataclass
class BiomeReference:
    """Reference to a biome file"""
    biome_id: str
    file_path: str
    enabled: bool = True


@dataclass
class World:
    """World configuration"""
    world_id: str
    name: str
    parameters: WorldParameters
    simulation: SimulationConfig
    biomes: List[BiomeReference]
    description: Optional[str] = None


# ============================================================================
# Knowledge Token Definitions
# ============================================================================

@dataclass
class TokenDecayConfig:
    """Decay configuration for a token type"""
    freshness_rate: float
    reliability_rate: float
    eviction_threshold: float = 0.0


@dataclass
class TokenMergeConfig:
    """Merge algorithm configuration"""
    algorithm: str  # weighted_average, most_recent, most_reliable, logical_or
    weight_freshness: float = 0.5


@dataclass
class TokenGossipConfig:
    """Gossip configuration for a token type"""
    shareable: bool = True
    attenuation: float = 0.1


@dataclass
class TokenInitialValues:
    """Initial values for new tokens"""
    freshness: float = 1.0
    reliability: float = 1.0
    source: str = "direct"


@dataclass
class TokenDefinition:
    """Knowledge token type definition"""
    kind: str
    description: str
    value_type: str  # float, boolean, position, entity_id
    decay: TokenDecayConfig
    merge: TokenMergeConfig
    gossip: TokenGossipConfig
    initial_values: TokenInitialValues
    value_range: Optional[Dict[str, float]] = None


# ============================================================================
# Interaction Rules
# ============================================================================

@dataclass
class InteractionParticipant:
    """Participant filter for interaction rule"""
    tags: List[str]
    species_id: Optional[str] = None


@dataclass
class InteractionTrigger:
    """Trigger conditions for interaction"""
    max_distance: Optional[float] = None
    requires_line_of_sight: bool = False


@dataclass
class InteractionEffects:
    """Effects of an interaction"""
    gossip: Optional[Dict[str, Any]] = None
    collision: Optional[Dict[str, Any]] = None
    territorial: Optional[Dict[str, Any]] = None


@dataclass
class InteractionRule:
    """Interaction rule between entities"""
    id: str
    type: str  # gossip, collision, territorial
    participants: Dict[str, InteractionParticipant]  # {entity_a, entity_b}
    trigger: InteractionTrigger
    effects: InteractionEffects


# ============================================================================
# Sensor Query Results (Phase 6)
# ============================================================================

@dataclass
class EntityHit:
    """Single entity detected by sensor query"""
    # Always present
    entity_id: str
    species_id: str

    # Optional fields (populated based on query flags)
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    pos_z: Optional[float] = None
    vel_x: Optional[float] = None
    vel_y: Optional[float] = None
    vel_z: Optional[float] = None
    distance: Optional[float] = None

    # Emission channels
    acoustic_amplitude: Optional[float] = None
    acoustic_peak_hz: Optional[float] = None
    bioluminescent_intensity: Optional[float] = None
    bioluminescent_wavelength_nm: Optional[float] = None
    optical_intensity: Optional[float] = None
    optical_wavelength_nm: Optional[float] = None
    optical_components: Optional[List[str]] = None
    thermal_temperature_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dict with only non-None fields.
        Ensures no numpy types leak through.
        """
        result = {
            'entity_id': self.entity_id,
            'species_id': self.species_id
        }

        # Add optional fields only if present
        optional_float_fields = [
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z',
            'distance',
            'acoustic_amplitude', 'acoustic_peak_hz',
            'bioluminescent_intensity', 'bioluminescent_wavelength_nm',
            'optical_intensity', 'optical_wavelength_nm',
            'thermal_temperature_delta'
        ]

        for field_name in optional_float_fields:
            value = getattr(self, field_name)
            if value is not None:
                # Cast to builtin float to avoid numpy types
                result[field_name] = float(value)

        # Add optical_components list if present
        if self.optical_components is not None:
            result['optical_components'] = self.optical_components

        return result


@dataclass
class SpatialQueryResult:
    """Result from spatial sensor query (e.g., query_cone)"""
    entities: List[EntityHit]
    timestamp: float
    query_origin: tuple  # (x, y, z) - tuple for serializability

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dict.
        Converts query_origin tuple to list, calls to_dict() on each EntityHit.
        """
        return {
            'entities': [hit.to_dict() for hit in self.entities],
            'timestamp': float(self.timestamp),
            'query_origin': list(self.query_origin)  # tuple -> list for JSON
        }
