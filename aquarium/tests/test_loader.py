"""
Test Phase 1: Data loading system

Verifies YAML → Python dataclass conversion works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aquarium.loader import (
    load_species, load_biome, load_world,
    load_token_definitions, load_interaction_rules,
    load_all_data
)


def test_load_species():
    """Test loading Drifter species"""
    data_root = Path(__file__).parent.parent.parent / "data"
    species_file = data_root / "species" / "sp-001-drifter.yaml"

    species = load_species(species_file)

    print(f"[OK] Loaded species: {species.name} ({species.species_id})")
    print(f"  Tags: {', '.join(species.tags)}")
    print(f"  Behaviors: {len(species.behaviors)}")
    print(f"  Max speed: {species.movement.max_speed_ms} m/s")
    print(f"  Gossip range: {species.knowledge.gossip_range} m")

    # Verify behaviors loaded correctly
    assert len(species.behaviors) == 4, "Expected 4 behaviors"
    assert species.behaviors[0].id == "flee-predator"
    assert species.behaviors[0].priority == 1

    print("[OK] Species behavioral rules loaded correctly\n")


def test_load_biome():
    """Test loading Vent Field Alpha biome"""
    data_root = Path(__file__).parent.parent.parent / "data"
    biome_file = data_root / "biomes" / "vent-field-alpha.yaml"

    biome = load_biome(biome_file)

    print(f"[OK] Loaded biome: {biome.name} ({biome.biome_id})")
    print(f"  Bounds: center={biome.bounds['center']}, radius={biome.bounds['radius']}m")
    print(f"  Obstacles: {len(biome.obstacles['spheres'])} vents, "
          f"{len(biome.obstacles['cylinders'])} ridges, "
          f"{len(biome.obstacles['planes'])} planes")

    if biome.spawning:
        total_entities = sum(s.count for s in biome.spawning['species'])
        print(f"  Spawning: {total_entities} total entities across {len(biome.spawning['species'])} species")

    # Verify obstacles
    assert len(biome.obstacles['spheres']) == 12, "Expected 12 vents"
    assert len(biome.obstacles['cylinders']) == 3, "Expected 3 ridges"
    assert len(biome.obstacles['planes']) == 1, "Expected 1 seabed"

    print("[OK] Biome obstacles and spawning loaded correctly\n")


def test_load_world():
    """Test loading Titan world config"""
    data_root = Path(__file__).parent.parent.parent / "data"
    world_file = data_root / "world" / "titan.yaml"

    world = load_world(world_file)

    print(f"[OK] Loaded world: {world.name} ({world.world_id})")
    print(f"  Gravity: {world.parameters.gravity_ms2} m/s²")
    print(f"  Fluid density: {world.parameters.fluid_density_kgm3} kg/m³")
    print(f"  Tick delta: {world.simulation.tick_delta_seconds} s")
    print(f"  Active radius: {world.simulation.active_region_radius} m")
    print(f"  Biomes: {len(world.biomes)}")

    # Verify simulation defaults
    assert world.simulation.influence_radius_factor == 2.5
    assert world.simulation.avoidance_weight == 0.6
    assert world.simulation.seabed_influence_distance == 5.0

    print("[OK] World parameters and simulation config loaded correctly\n")


def test_load_tokens():
    """Test loading knowledge token definitions"""
    data_root = Path(__file__).parent.parent.parent / "data"
    tokens_file = data_root / "knowledge" / "tokens.yaml"

    tokens = load_token_definitions(tokens_file)

    print(f"[OK] Loaded {len(tokens)} knowledge token types:")
    for kind, token_def in tokens.items():
        print(f"  - {kind}: {token_def.value_type}, "
              f"freshness_decay={token_def.decay.freshness_rate}/s, "
              f"merge={token_def.merge.algorithm}")

    # Verify key tokens exist
    assert 'ship_sentiment' in tokens
    assert 'predator_location' in tokens
    assert tokens['ship_sentiment'].value_type == 'float'

    print("[OK] Knowledge token definitions loaded correctly\n")


def test_load_interactions():
    """Test loading interaction rules"""
    data_root = Path(__file__).parent.parent.parent / "data"
    interactions_file = data_root / "interactions" / "rules.yaml"

    rules = load_interaction_rules(interactions_file)

    print(f"[OK] Loaded {len(rules)} interaction rules:")
    for rule in rules:
        print(f"  - {rule.id}: {rule.type}")

    # Verify gossip rule exists
    gossip_rules = [r for r in rules if r.type == 'gossip']
    assert len(gossip_rules) > 0, "Expected at least one gossip rule"

    print("[OK] Interaction rules loaded correctly\n")


def test_load_all():
    """Test loading entire data pack"""
    data_root = Path(__file__).parent.parent.parent / "data"
    schema_dir = data_root / "schemas"

    print("Loading complete data pack...")
    data = load_all_data(data_root, schema_dir)

    print(f"\n[OK] Complete data pack loaded:")
    print(f"  World: {data['world'].name}")
    print(f"  Species: {len(data['species'])}")
    print(f"  Biomes: {len(data['biomes'])}")
    print(f"  Token types: {len(data['tokens'])}")
    print(f"  Interaction rules: {len(data['interactions'])}")

    print("\n[PASS] Phase 1 Complete: Data loading system works!")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 1 Test: Data Loading System")
    print("=" * 60)
    print()

    try:
        test_load_species()
        test_load_biome()
        test_load_world()
        test_load_tokens()
        test_load_interactions()
        test_load_all()

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All Phase 1 tests passed!")
    print("=" * 60)
