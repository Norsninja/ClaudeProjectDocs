"""
Synthetic performance test harness.

Creates controlled entity populations for performance testing without modifying biome data.
Per SD guidance: deterministic seeded spawns with realistic tag distributions.
"""

import numpy as np
from typing import List, Dict
from pathlib import Path

from aquarium.simulation import AquariumSimulation
from aquarium.entity import Entity


def build_perf_scenario(
    data_root: Path,
    N: int,
    predator_ratio: float = 0.01,
    ship_count: int = 1,
    seed: int = 42
) -> AquariumSimulation:
    """
    Build synthetic performance scenario with N entities.

    Entity distribution (per SD guidance):
    - ~90% drifters (passive, no predator tag)
    - ~9% social entities (for gossip testing later)
    - ~1% predators
    - 1 ship

    Args:
        data_root: Path to data directory
        N: Total entity count
        predator_ratio: Fraction of entities that are predators (default 0.01 = 1%)
        ship_count: Number of ships (default 1)
        seed: Random seed for deterministic placement

    Returns:
        Simulation with N synthetic entities in first biome
    """
    # Create base simulation (empty or minimal spawn)
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=0,  # Don't spawn from biome configs
        spawn_only_species=None
    )

    # Get first biome for placement
    biome_id = list(sim.biomes.keys())[0]
    biome = sim.biomes[biome_id]
    center = np.array(biome.bounds['center'], dtype=np.float64)
    radius = biome.bounds['radius']

    # Calculate entity counts per category
    n_predators = max(1, int(N * predator_ratio))
    n_ship = ship_count
    n_drifters = N - n_predators - n_ship

    # Seed RNG for deterministic placement
    rng = np.random.RandomState(seed)

    print(f"[Perf Harness] Creating {N} synthetic entities:")
    print(f"  - {n_drifters} drifters")
    print(f"  - {n_predators} predators")
    print(f"  - {n_ship} ship(s)")

    # Helper: Generate random position within biome sphere
    def random_position_in_sphere(rng, center, radius, count):
        """Generate uniformly distributed points within sphere."""
        positions = []
        for _ in range(count):
            # Rejection sampling for uniform sphere
            while True:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    positions.append(center + p)
                    break
        return positions

    # Generate positions
    drifter_positions = random_position_in_sphere(rng, center, radius, n_drifters)
    predator_positions = random_position_in_sphere(rng, center, radius, n_predators)
    ship_positions = random_position_in_sphere(rng, center, radius, n_ship)

    # Create entities
    synthetic_entities = []
    entity_count = 0

    # Drifters (no special tags)
    for i, pos in enumerate(drifter_positions):
        entity = Entity(
            instance_id=f"perf-drifter-{entity_count:04d}",
            species_id="sp-001-drifter",
            biome_id=biome_id,
            position=np.array(pos, dtype=np.float64),
            velocity=np.array([0.1, 0.0, 0.1], dtype=np.float64),  # Small drift
            size_factor=1.0,
            tags=['mobile', 'drifter']
        )
        synthetic_entities.append(entity)
        entity_count += 1

    # Predators (with 'predator' tag)
    for i, pos in enumerate(predator_positions):
        entity = Entity(
            instance_id=f"perf-predator-{entity_count:04d}",
            species_id="sp-003-shadow",
            biome_id=biome_id,
            position=np.array(pos, dtype=np.float64),
            velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            size_factor=1.5,
            tags=['mobile', 'predator']
        )
        synthetic_entities.append(entity)
        entity_count += 1

    # Ships (with 'ship' tag)
    for i, pos in enumerate(ship_positions):
        entity = Entity(
            instance_id=f"perf-ship-{entity_count:04d}",
            species_id="ship",
            biome_id=biome_id,
            position=np.array(pos, dtype=np.float64),
            velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            size_factor=2.0,
            tags=['ship']
        )
        synthetic_entities.append(entity)
        entity_count += 1

    # Replace simulation entities with synthetic population
    sim.entities = synthetic_entities

    # Initialize spatial structures for new entity count
    sim._ensure_position_capacity(len(sim.entities))

    print(f"[OK] Created {len(sim.entities)} synthetic entities in {biome.name}")

    return sim


def get_tag_distribution(entities: List[Entity]) -> Dict[str, int]:
    """Get count of entities per tag for validation."""
    tag_counts = {}
    for entity in entities:
        for tag in entity.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return tag_counts


if __name__ == "__main__":
    # Quick test
    data_root = Path(__file__).parent.parent.parent / "data"
    sim = build_perf_scenario(data_root, N=1000, predator_ratio=0.01, ship_count=1, seed=42)

    print(f"\n[Test] Tag distribution:")
    tag_dist = get_tag_distribution(sim.entities)
    for tag, count in sorted(tag_dist.items()):
        print(f"  {tag}: {count}")

    print(f"\n[Test] Running 10 ticks...")
    for i in range(10):
        sim.tick()

    stats = sim.get_tick_stats()
    print(f"[OK] {stats['tick_count']} ticks completed")
    print(f"  Avg tick: {stats['avg_tick_time_ms']:.3f} ms")
