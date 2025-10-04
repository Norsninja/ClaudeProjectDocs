"""
Central configuration constants for aquarium simulation.

Defines default values, thresholds, and configuration parameters
used across multiple modules.
"""

# ============================================================================
# Spatial Indexing Configuration
# ============================================================================

# Enable scipy.cKDTree spatial indexing (Phase 4+)
# Set to False to use O(n) fallback for performance comparison
USE_CKDTREE = True

# cKDTree build parameters (Phase 4+)
CKDTREE_LEAFSIZE = 16  # Leaf size for cKDTree construction
CKDTREE_WORKERS = 1   # Single-threaded (1.86x faster than -1 for gossip @ 1000 entities)

# Phase 5: Per-tag, per-biome subtrees for batch queries
USE_PER_TAG_TREES = True  # Enable subtree optimization (Phase 5+)

# Phase 5: Batch query implementation (vectorized vs legacy)
USE_LEGACY_BATCH = False  # Use legacy dict-based batch queries (for A/B testing)

# Phase 5: Nearest entity query distances (per SD guidance)
NEAREST_PREDATOR_RADIUS = 80.0   # Max distance for predator detection (meters)
NEAREST_SHIP_RADIUS = 150.0      # Max distance for ship detection (meters)


# ============================================================================
# Obstacle Avoidance Configuration (Phase 4)
# ============================================================================

# Lookahead sampling distances for collision prediction
AVOIDANCE_LOOKAHEAD_DISTANCES = [0.5, 2.0]  # meters ahead to sample

# Default blend weight for avoidance force (0.0 = no avoidance, 1.0 = full avoidance)
AVOIDANCE_WEIGHT_DEFAULT = 0.6

# Default influence radius multiplier when obstacle doesn't specify override
INFLUENCE_RADIUS_FACTOR_DEFAULT = 2.5

# Optional acceleration clamp to prevent sharp velocity changes (None = disabled)
ACCELERATION_CLAMP_MS2 = None  # m/s^2, off by default

# Performance timing breakdown for avoidance (default False for zero overhead)
AVOIDANCE_TIMING_BREAKDOWN = False  # Set True to measure sphere/cylinder/plane separately


# ============================================================================
# Sensor Query Emission Defaults (Phase 6)
# ============================================================================

# Fallback emission values when Species.emissions fields are missing
ACOUSTIC_DEFAULT_AMPLITUDE = 0.1      # Normalized amplitude (0.0-1.0)
ACOUSTIC_DEFAULT_PEAK_HZ = 50.0       # Low-frequency default (Hz)
BIOLUM_DEFAULT_INTENSITY = 0.05       # Dim default (0.0-1.0)
BIOLUM_DEFAULT_WAVELENGTH = 480.0     # Blue-green wavelength (nm)
VENT_THERMAL_BASE_DELTA = 5.0         # Celsius above ambient for vents


# ============================================================================
# Knowledge Token Defaults
# ============================================================================

# Default values for knowledge tokens when not present in entity
TOKEN_DEFAULTS = {
    'ship_sentiment': 0.0,  # Neutral sentiment (range: -1.0 hostile to 1.0 friendly)
}


# ============================================================================
# Knowledge Gossip Configuration (Phase 6+)
# ============================================================================

# Enable knowledge gossip system
USE_GOSSIP = True

# Maximum token exchanges per entity per tick
GOSSIP_EXCHANGES_PER_ENTITY = 2

# Fallback gossip range when Species.gossip_range_m is not specified
GOSSIP_FALLBACK_RANGE_M = 15.0

# Allowed token kinds for gossip (Phase 2a+: multi-kind support)
GOSSIP_ALLOWED_KINDS = ['ship_sentiment', 'predator_location']

# Minimum gossip neighbors threshold for diagnostics (warn if entity has fewer)
MIN_GOSSIP_NEIGHBORS = 3

# Token capacity per entity (Phase 2: lifecycle management)
GOSSIP_TOKEN_CAP_DEFAULT = 16  # Max tokens per entity across all kinds

# Logging interval for gossip network telemetry (ticks)
GOSSIP_LOG_INTERVAL = 100  # Print degree histogram every 100 ticks


# ============================================================================
# Spawning Configuration
# ============================================================================

# Initial velocity as fraction of max_speed_ms for spawned entities
CRUISE_SPEED_FRACTION = 0.2  # 20% of max speed (used only for initial drift)


# ============================================================================
# Performance Configuration
# ============================================================================

# Tick timing window for rolling average
TICK_TIME_WINDOW = 100  # Number of ticks to average

# Default tick summary interval (print every N ticks)
TICK_SUMMARY_INTERVAL = 100  # Print summary every 100 ticks


# ============================================================================
# Ecosystem & Energy Configuration (Phase 8+)
# ============================================================================

# Enable ecosystem energy/feeding/death systems
USE_ECOSYSTEM = True  # Feature toggle for energy/feeding/death

# Resource field defaults (when biome/vent lacks specific values)
RESOURCE_DEFAULT_PEAK = 100.0  # Peak plankton density at vent (units/m³)
RESOURCE_DEFAULT_SIGMA = 50.0  # Gaussian falloff distance (meters)
RESOURCE_DEFAULT_BASE = 0.0  # Background density everywhere (0.0 = no ambient food)

# Metabolism defaults (when Species.metabolism not specified)
ENERGY_MAX_DEFAULT = 100.0  # Default maximum energy capacity
ENERGY_DRAIN_DEFAULT = 0.5  # Passive drain per tick
MOVEMENT_COST_FACTOR_DEFAULT = 0.1  # Energy per m/s velocity
FEEDING_EFFICIENCY_DEFAULT = 2.0  # Energy gained per unit resource consumed

# Feeding defaults (when Species.feeding not specified)
INTAKE_RATE_DEFAULT = 1.0  # Max units of resource consumed per tick
FEEDING_COOLDOWN_TICKS_DEFAULT = 5  # Ticks between feeding actions

# Logging interval for ecosystem telemetry (ticks)
ECOSYSTEM_LOG_INTERVAL = 100  # Print population/energy stats every 100 ticks
