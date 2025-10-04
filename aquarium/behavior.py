"""
Behavior evaluation engine for entity AI.

Evaluates behavior priority lists, checks conditions, and maps actions to velocity.
Phase 3: Simple O(n) neighbor searches (defer cKDTree to Phase 4).
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from .entity import Entity
from .data_types import Species, Behavior, BehaviorCondition, BehaviorAction
from .spatial import normalize, clamp_speed
from .constants import TOKEN_DEFAULTS

if TYPE_CHECKING:
    from .spatial_queries import SpatialIndexAdapter


def _get_nearest_from_cache(
    entity: Entity,
    tag: str,
    spatial: 'SpatialIndexAdapter',
    query_cache: dict,
    all_entities: List[Entity]
) -> Optional[Entity]:
    """
    Get nearest entity with tag from cache, or fallback to per-entity query.

    Phase 5 helper function to abstract cache lookup pattern.

    Args:
        entity: Source entity
        tag: Tag to search for (e.g., 'predator', 'ship')
        spatial: Spatial index adapter
        query_cache: Batch query cache (may be None)
        all_entities: All entities (for fallback)

    Returns:
        Nearest entity with tag, or None if not found
    """
    # Phase 5 refined: Try array-based cache first
    if query_cache is not None and tag in query_cache.get('nearest', {}):
        entity_row = spatial._row_of_id.get(entity.instance_id)
        if entity_row is not None:
            # Phase 5 refined: Array indexing (no dict lookup)
            dist_arr, idx_arr = query_cache['nearest'][tag]
            nearest_row = idx_arr[entity_row]
            dist = dist_arr[entity_row]

            # Check if match found
            if nearest_row != -1 and dist != np.inf:
                return all_entities[nearest_row]

    # Fallback to per-entity query
    return spatial.find_nearest_by_tag(entity, tag, max_distance=None)


def evaluate_behavior(
    entity: Entity,
    species: Species,
    all_entities: List[Entity],
    max_speed: float,
    spatial: 'SpatialIndexAdapter',
    query_cache: dict = None,
    behavior_context: Optional[dict] = None
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    """
    Evaluate entity behavior and compute resulting velocity.

    Processes behaviors in priority order, stops at first match.

    Args:
        entity: Entity to evaluate
        species: Species definition with behaviors
        all_entities: All entities in same biome (for neighbor search)
        max_speed: Species max_speed_ms (for speed multiplier calculation)
        spatial: Spatial index adapter for queries
        query_cache: Phase 5 batch query cache (optional)
        behavior_context: Phase 8 context dict (e.g., {'energy_view': EnergyView})

    Returns:
        Tuple of (behavior_id, velocity, emission_multipliers)

    Example:
        behavior_id, velocity, emissions = evaluate_behavior(entity, species, entities, 10.0, spatial, cache)
    """
    # Sort behaviors by priority (ascending: 1, 2, 3, ...)
    sorted_behaviors = sorted(species.behaviors, key=lambda b: b.priority)

    # Evaluate each behavior until one matches
    for behavior in sorted_behaviors:
        if _check_conditions(entity, behavior.conditions, all_entities, spatial, query_cache, behavior_context):
            # Conditions met, apply action
            velocity = _apply_action(entity, behavior.action, all_entities, max_speed, spatial, query_cache)
            emissions = behavior.action.emission_multipliers or {}

            return behavior.id, velocity, emissions

    # No behavior matched (shouldn't happen if fallback exists)
    # Return current velocity unchanged
    return None, entity.velocity.copy(), {}


def _check_conditions(
    entity: Entity,
    conditions: List[BehaviorCondition],
    all_entities: List[Entity],
    spatial: 'SpatialIndexAdapter',
    query_cache: dict = None,
    behavior_context: Optional[dict] = None
) -> bool:
    """
    Check if all conditions are met (AND logic).

    Args:
        entity: Entity being evaluated
        conditions: List of conditions to check
        all_entities: All entities (for neighbor searches)
        spatial: Spatial index adapter for queries
        query_cache: Phase 5 batch query cache (optional)
        behavior_context: Phase 8 context dict (e.g., {'energy_view': EnergyView})

    Returns:
        True if all conditions met, False otherwise
    """
    # Empty conditions = always true (fallback behavior)
    if not conditions:
        return True

    # All conditions must be true (AND)
    for condition in conditions:
        if not _check_single_condition(entity, condition, all_entities, spatial, query_cache, behavior_context):
            return False

    return True


def _check_single_condition(
    entity: Entity,
    condition: BehaviorCondition,
    all_entities: List[Entity],
    spatial: 'SpatialIndexAdapter',
    query_cache: dict = None,
    behavior_context: Optional[dict] = None
) -> bool:
    """
    Check a single condition.

    Supported types:
    - nearest_entity: Search by tag, check distance threshold
    - knowledge_token: Compare token value
    - token_exists: Check if token present
    - within_depth_band: Check if y-coordinate within range
    - energy_below: Check if current energy < threshold (Phase 8)

    Args:
        entity: Entity being evaluated
        condition: Condition to check
        all_entities: All entities (for nearest_entity search)
        spatial: Spatial index adapter for queries
        query_cache: Phase 5 batch query cache (optional)
        behavior_context: Phase 8 context dict (e.g., {'energy_view': EnergyView})

    Returns:
        True if condition met, False otherwise
    """
    if condition.type == "nearest_entity":
        # Phase 5 refined: Check array-based cache first (if available)
        if query_cache is not None:
            tag = condition.tag
            if tag in query_cache.get('nearest', {}):
                # Get entity row from spatial adapter
                entity_row = spatial._row_of_id.get(entity.instance_id)
                if entity_row is not None:
                    # Phase 5 refined: Array indexing (no dict lookup)
                    dist_arr, idx_arr = query_cache['nearest'][tag]
                    nearest_row = idx_arr[entity_row]
                    dist = dist_arr[entity_row]

                    # Check if match found in cache
                    if nearest_row != -1 and dist != np.inf:
                        # Cache hit: check max_distance constraint (condition-specific)
                        if condition.max_distance is not None and dist > condition.max_distance:
                            return False

                        # Match found within distance
                        return True

                    # Cache reported no match: fall through to per-entity query
                    # This handles cases where condition.max_distance > cache radius

        # Fallback to per-entity query (cache miss, no match, or disabled)
        # Uses condition.max_distance for accurate range checking
        target = spatial.find_nearest_by_tag(entity, condition.tag, max_distance=condition.max_distance)

        if target is None:
            return False

        # Target found within max_distance
        return True

    elif condition.type == "knowledge_token":
        # Get token value (or default)
        token_kind = condition.token
        token_value = entity.knowledge_tokens.get(token_kind)

        if token_value is None:
            # Use default if available
            if token_kind in TOKEN_DEFAULTS:
                token_value = TOKEN_DEFAULTS[token_kind]
            else:
                return False

        # Phase 8: Handle gossip v2 dict tokens {value, version, timestamp}
        if isinstance(token_value, dict):
            token_value = token_value.get('value', TOKEN_DEFAULTS.get(token_kind, 0.0))

        # Compare value using operator
        operator = condition.operator
        threshold = condition.value

        if operator == "less_than":
            return token_value < threshold
        elif operator == "greater_than":
            return token_value > threshold
        elif operator == "greater_equal":
            return token_value >= threshold
        elif operator == "less_equal":
            return token_value <= threshold
        elif operator == "equal":
            return abs(token_value - threshold) < 1e-9
        else:
            return False

    elif condition.type == "token_exists":
        # Check if token present
        return condition.token in entity.knowledge_tokens

    elif condition.type == "within_depth_band":
        # Check y-coordinate (depth)
        y = entity.position[1]
        return condition.min_depth <= y <= condition.max_depth

    elif condition.type == "energy_below":
        # Phase 8: Check if current energy < threshold (hunger condition)
        if behavior_context is None or 'energy_view' not in behavior_context:
            # No context available - condition cannot be evaluated
            return False

        energy_view = behavior_context['energy_view']
        current_energy = energy_view.get_energy(entity.instance_id)
        threshold = condition.value

        return current_energy < threshold

    else:
        # Unknown condition type
        return False


def _apply_action(
    entity: Entity,
    action: BehaviorAction,
    all_entities: List[Entity],
    max_speed: float,
    spatial: 'SpatialIndexAdapter',
    query_cache: dict = None
) -> np.ndarray:
    """
    Map behavior action to velocity vector.

    Supported actions:
    - flee: Move away from nearest threat
    - investigate: Move toward target
    - forage: Maintain current drift velocity
    - hold: Zero velocity
    - align_with_current: Align with global current (Phase 4+)

    Args:
        entity: Entity performing action
        action: Action specification
        all_entities: All entities (for target finding)
        max_speed: Species max_speed_ms
        spatial: Spatial index adapter for queries
        query_cache: Phase 5 batch query cache (optional)

    Returns:
        Velocity vector [vx, vy, vz]
    """
    action_type = action.type
    speed_multiplier = action.speed_multiplier
    target_speed = max_speed * speed_multiplier

    if action_type == "flee":
        # Phase 5: Get nearest predator/ship from cache (or fallback)
        predator = _get_nearest_from_cache(entity, "predator", spatial, query_cache, all_entities)
        ship = _get_nearest_from_cache(entity, "ship", spatial, query_cache, all_entities)

        # Choose closer threat
        threat = None
        if predator and ship:
            dist_predator = np.linalg.norm(entity.position - predator.position)
            dist_ship = np.linalg.norm(entity.position - ship.position)
            threat = predator if dist_predator < dist_ship else ship
        elif predator:
            threat = predator
        elif ship:
            threat = ship

        if threat:
            # Move away from threat
            direction = entity.position - threat.position
            direction, _ = normalize(direction)
            return direction * target_speed
        else:
            # No threat found, maintain current velocity
            return clamp_speed(entity.velocity, target_speed)

    elif action_type == "investigate":
        # Phase 5: Get nearest ship from cache (or fallback)
        ship = _get_nearest_from_cache(entity, "ship", spatial, query_cache, all_entities)

        if ship:
            direction = ship.position - entity.position
            direction, _ = normalize(direction)
            return direction * target_speed
        else:
            # No ship, maintain current velocity
            return clamp_speed(entity.velocity, target_speed)

    elif action_type == "forage":
        # Maintain current drift velocity (clamp to target speed)
        return clamp_speed(entity.velocity, target_speed)

    elif action_type == "hold":
        # Zero velocity (stationary)
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    elif action_type == "align_with_current":
        # Phase 4+: Align with global current
        # For now, maintain drift velocity
        return clamp_speed(entity.velocity, target_speed)

    else:
        # Unknown action, maintain velocity
        return entity.velocity.copy()


def update_entity_behavior(
    entity: Entity,
    species: Species,
    all_entities: List[Entity],
    spatial: 'SpatialIndexAdapter',
    query_cache: dict = None,
    behavior_context: Optional[dict] = None
):
    """
    Evaluate and update entity behavior state.

    Updates entity.active_behavior_id, entity.velocity, entity.emission_multipliers.

    Args:
        entity: Entity to update
        species: Species definition
        all_entities: All entities in biome
        spatial: Spatial index adapter for queries
        query_cache: Phase 5 batch query cache (optional, falls back to per-entity queries)
        behavior_context: Phase 8 context dict (e.g., {'energy_view': EnergyView})
    """
    # Evaluate behavior
    behavior_id, velocity, emissions = evaluate_behavior(
        entity=entity,
        species=species,
        all_entities=all_entities,
        max_speed=species.movement.max_speed_ms,
        spatial=spatial,
        query_cache=query_cache,
        behavior_context=behavior_context
    )

    # Update entity state
    entity.active_behavior_id = behavior_id
    entity.velocity = velocity

    # Update emission multipliers (merge with defaults)
    for channel, multiplier in emissions.items():
        entity.emission_multipliers[channel] = multiplier
