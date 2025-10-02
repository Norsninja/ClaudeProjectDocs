"""
YAML data loader with schema validation.

Loads species, biomes, world, interactions, and knowledge token definitions
from YAML files and validates against JSON schemas.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
import jsonschema

from .data_types import (
    Species, Biome, World, TokenDefinition, InteractionRule,
    PhysicalProperties, EmissionProfile, MovementProperties,
    Behavior, BehaviorCondition, BehaviorAction, KnowledgeConfig,
    SphereObstacle, CylinderObstacle, PlaneObstacle, SpawningConfig,
    WorldParameters, SimulationConfig, BiomeReference,
    TokenDecayConfig, TokenMergeConfig, TokenGossipConfig, TokenInitialValues,
    InteractionParticipant, InteractionTrigger, InteractionEffects
)


class DataLoadError(Exception):
    """Raised when data loading or validation fails"""
    pass


def load_yaml(file_path: Path) -> dict:
    """Load YAML file and return parsed dict"""
    if not file_path.exists():
        raise DataLoadError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DataLoadError(f"YAML parse error in {file_path}: {e}")


def validate_against_schema(data: dict, schema_path: Path, data_path: Path):
    """Validate data dict against JSON schema"""
    if not schema_path.exists():
        # Schema validation optional (for MVP, schemas may not exist yet)
        return

    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise DataLoadError(f"Validation error in {data_path}: {e.message}")
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON schema {schema_path}: {e}")


def load_species(file_path: Path, schema_dir: Optional[Path] = None) -> Species:
    """Load species definition from YAML"""
    data = load_yaml(file_path)

    # Validate if schema available
    if schema_dir:
        schema_path = schema_dir / "species.schema.json"
        validate_against_schema(data, schema_path, file_path)

    # Parse into dataclasses
    physical = PhysicalProperties(**data['physical'])
    emissions = EmissionProfile(**data.get('emissions', {}))
    movement = MovementProperties(**data['movement'])

    behaviors = []
    for b_data in data['behaviors']:
        conditions = [BehaviorCondition(**c) for c in b_data['conditions']]
        action = BehaviorAction(**b_data['action'])
        behaviors.append(Behavior(
            id=b_data['id'],
            priority=b_data['priority'],
            conditions=conditions,
            action=action
        ))

    knowledge = KnowledgeConfig(**data.get('knowledge', {}))

    return Species(
        species_id=data['species_id'],
        name=data['name'],
        tags=data['tags'],
        physical=physical,
        emissions=emissions,
        movement=movement,
        behaviors=behaviors,
        knowledge=knowledge,
        parameters=data.get('parameters', {}),
        description=data.get('description')
    )


def load_biome(file_path: Path, schema_dir: Optional[Path] = None) -> Biome:
    """Load biome definition from YAML"""
    data = load_yaml(file_path)

    # Validate if schema available
    if schema_dir:
        schema_path = schema_dir / "biome.schema.json"
        validate_against_schema(data, schema_path, file_path)

    # Parse obstacles
    obstacles = {
        'spheres': [],
        'cylinders': [],
        'planes': []
    }

    obstacle_data = data.get('obstacles', {})

    for sphere_data in obstacle_data.get('spheres', []):
        obstacles['spheres'].append(SphereObstacle(**sphere_data))

    for cylinder_data in obstacle_data.get('cylinders', []):
        obstacles['cylinders'].append(CylinderObstacle(**cylinder_data))

    for plane_data in obstacle_data.get('planes', []):
        obstacles['planes'].append(PlaneObstacle(**plane_data))

    # Parse spawning config
    spawning = None
    if 'spawning' in data:
        spawning = {
            'species': [SpawningConfig(**s) for s in data['spawning'].get('species', [])]
        }

    return Biome(
        biome_id=data['biome_id'],
        name=data['name'],
        bounds=data['bounds'],
        obstacles=obstacles,
        spawning=spawning,
        seed=data.get('seed'),
        description=data.get('description')
    )


def load_world(file_path: Path, schema_dir: Optional[Path] = None) -> World:
    """Load world configuration from YAML"""
    data = load_yaml(file_path)

    # Validate if schema available
    if schema_dir:
        schema_path = schema_dir / "world.schema.json"
        validate_against_schema(data, schema_path, file_path)

    parameters = WorldParameters(**data['parameters'])
    simulation = SimulationConfig(**data.get('simulation', {}))
    biomes = [BiomeReference(**b) for b in data['biomes']]

    return World(
        world_id=data['world_id'],
        name=data['name'],
        parameters=parameters,
        simulation=simulation,
        biomes=biomes,
        description=data.get('description')
    )


def load_token_definitions(file_path: Path, schema_dir: Optional[Path] = None) -> Dict[str, TokenDefinition]:
    """Load knowledge token definitions from YAML"""
    data = load_yaml(file_path)

    # Validate if schema available
    if schema_dir:
        schema_path = schema_dir / "knowledge_token.schema.json"
        validate_against_schema(data, schema_path, file_path)

    tokens = {}
    for token_data in data['tokens']:
        decay = TokenDecayConfig(**token_data['decay'])
        merge = TokenMergeConfig(**token_data['merge'])
        gossip = TokenGossipConfig(**token_data['gossip'])
        initial = TokenInitialValues(**token_data.get('initial_values', {}))

        token_def = TokenDefinition(
            kind=token_data['kind'],
            description=token_data['description'],
            value_type=token_data['value_type'],
            decay=decay,
            merge=merge,
            gossip=gossip,
            initial_values=initial,
            value_range=token_data.get('value_range')
        )
        tokens[token_def.kind] = token_def

    return tokens


def load_interaction_rules(file_path: Path, schema_dir: Optional[Path] = None) -> List[InteractionRule]:
    """Load interaction rules from YAML"""
    data = load_yaml(file_path)

    # Validate if schema available
    if schema_dir:
        schema_path = schema_dir / "interaction.schema.json"
        validate_against_schema(data, schema_path, file_path)

    rules = []
    for rule_data in data['rules']:
        participants = {
            key: InteractionParticipant(**val)
            for key, val in rule_data['participants'].items()
        }
        trigger = InteractionTrigger(**rule_data.get('trigger', {}))
        effects = InteractionEffects(**rule_data.get('effects', {}))

        rules.append(InteractionRule(
            id=rule_data['id'],
            type=rule_data['type'],
            participants=participants,
            trigger=trigger,
            effects=effects
        ))

    return rules


def load_species_registry(species_dir: Path, schema_dir: Optional[Path] = None) -> Dict[str, Species]:
    """Load all species from directory"""
    species_dir = Path(species_dir)
    if not species_dir.exists():
        raise DataLoadError(f"Species directory not found: {species_dir}")

    registry = {}
    for yaml_file in species_dir.glob("*.yaml"):
        species = load_species(yaml_file, schema_dir)
        registry[species.species_id] = species

    if not registry:
        raise DataLoadError(f"No species files found in {species_dir}")

    return registry


def load_all_data(data_root: Path, schema_dir: Optional[Path] = None) -> dict:
    """Load all simulation data from data directory

    Returns dict with keys: world, biomes, species, tokens, interactions
    """
    data_root = Path(data_root)

    # Load world
    world = load_world(data_root / "world" / "titan.yaml", schema_dir)

    # Load species
    species = load_species_registry(data_root / "species", schema_dir)

    # Load biomes (referenced in world config)
    biomes = {}
    for biome_ref in world.biomes:
        if biome_ref.enabled:
            biome_path = data_root / biome_ref.file_path
            biomes[biome_ref.biome_id] = load_biome(biome_path, schema_dir)

    # Load knowledge tokens
    tokens = load_token_definitions(data_root / "knowledge" / "tokens.yaml", schema_dir)

    # Load interactions
    interactions = load_interaction_rules(data_root / "interactions" / "rules.yaml", schema_dir)

    return {
        'world': world,
        'biomes': biomes,
        'species': species,
        'tokens': tokens,
        'interactions': interactions
    }
