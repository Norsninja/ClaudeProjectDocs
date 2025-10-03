"""
Knowledge gossip system for aquarium simulation.

Entities exchange knowledge tokens (ship_sentiment, predator_location, etc.)
with nearby entities. Fully vectorized implementation with array-based token
state and merge operations.

Algorithm v2 (2025-10-02):
- Radius-based neighbors (bidirectional, per-species gossip_range_m)
- Push-pull hybrid protocol (symmetric exchange per undirected edge)
- Version + timestamp propagation (no attenuation in Phase 1)
- float64 precision (no float32 errors)

Performance: O(N·log N) for spatial index + O(E) for edge merges, all vectorized.
Target: <2ms @ 1000 entities.

Phase 1: Exchange only (version-based precedence, no decay/eviction)
Phase 2: Lifecycle (time-based decay, eviction, capacity enforcement, attenuation)

Design: project/plans/gossip_v2_phase1_design.md
Research: project/research/spatial_gossip_protocols_external_research_2025-10-02.md
"""

import numpy as np
import time
from typing import List, Dict, Optional
from collections import defaultdict
from contextlib import contextmanager
import yaml
import os

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from aquarium.entity import Entity
from aquarium.spatial_queries import SpatialIndexAdapter
from aquarium.constants import (
    GOSSIP_EXCHANGES_PER_ENTITY,
    GOSSIP_FALLBACK_RANGE_M,
    GOSSIP_ALLOWED_KINDS,
    MIN_GOSSIP_NEIGHBORS,
    CKDTREE_WORKERS
)


# Global token definitions (loaded at module import, not per-call)
def _load_token_definitions_at_import() -> Dict:
    """
    Load token definitions from data/knowledge/tokens.yaml at module import.

    PERFORMANCE FIX: Called once during module initialization to avoid per-call
    YAML parsing overhead (~21ms first call).

    Returns:
        Dict mapping token kind to definition (decay, attenuation, etc.)
    """
    # Find tokens.yaml relative to this file
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    tokens_path = os.path.join(project_root, 'data', 'knowledge', 'tokens.yaml')

    with open(tokens_path, 'r') as f:
        data = yaml.safe_load(f)

    # Build lookup dict keyed by kind
    token_defs = {}
    for token_def in data['tokens']:
        kind = token_def['kind']
        token_defs[kind] = token_def

    return token_defs


# Load at module import (one-time cost, not measured in gossip perf tests)
_TOKEN_DEFINITIONS = _load_token_definitions_at_import()


# Module-level cache for per-row gossip ranges (keyed by adapter + build_seq)
# Cache entry: {'build_seq': int, 'range_by_row': np.ndarray, 'uniform': bool, 'max_range': float}
_RANGE_CACHE: Dict[int, Dict] = {}  # keyed by id(adapter)

# Module-level SoA cache for token state (keyed by adapter + build_seq)
# Cache entry: {'build_seq': int, 'N': int, 'has': bool[N], 'val': float64[N],
#               'version': int64[N], 'last_tick': float64[N], 'reliability': float64[N]}
# Eliminates per-tick entity loop on steady state (cache hit)
_TOKEN_STATE_CACHE: Dict[int, Dict] = {}  # keyed by id(adapter)


# Performance profiling support (enabled via GOSSIP_PROFILE=1)
class _GossipTimer:
    """
    Lightweight timer for profiling gossip sub-phases.

    Enabled when GOSSIP_PROFILE=1 environment variable is set.
    Adds negligible overhead when disabled (~1 branch per section).
    """
    def __init__(self):
        self.enabled = os.getenv('GOSSIP_PROFILE') == '1'
        self.timings = {}

    @contextmanager
    def time(self, name: str):
        """Context manager to time a code section."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter_ns()
        yield
        elapsed_ns = time.perf_counter_ns() - start
        elapsed_ms = elapsed_ns / 1_000_000
        # Accumulate times for repeated phases (e.g., per-kind operations)
        self.timings[name] = self.timings.get(name, 0.0) + elapsed_ms

    def reset(self):
        """Clear timings (called at start of each exchange_tokens call)."""
        self.timings.clear()


def _process_kind(
    kind: str,
    token_def: Dict,
    entities: List[Entity],
    adapter: SpatialIndexAdapter,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    entity_degrees: np.ndarray,
    current_tick: int,
    epsilon: float,
    timer: _GossipTimer
) -> tuple:
    """
    Process gossip for a single token kind.

    Phases:
    1. Extract: Build SoA arrays from entity.knowledge_tokens[kind]
    2. Decay: Compute freshness and reliability_current from age
    3. Merge: Vectorized push-pull using edge_a/edge_b
    4. Evict: Remove stale tokens below eviction_threshold
    5. Writeback: Update entity.knowledge_tokens[kind]

    Args:
        kind: Token kind to process (e.g., 'ship_sentiment')
        token_def: Token definition dict from _TOKEN_DEFINITIONS[kind]
        entities: Entity list (used via adapter._entities)
        adapter: SpatialIndexAdapter (for cache keys and _entities)
        edge_a, edge_b: Precomputed edge arrays (reused across kinds)
        entity_degrees: Neighbor counts (for diagnostics, unused in Phase 1)
        current_tick: Current simulation tick
        epsilon: Float comparison epsilon (1e-12)
        timer: Profiling timer context

    Returns:
        Tuple of (exchanges_count, changed_rows, profile_dict):
            - exchanges_count (int): Number of token transfers (push + pull)
            - changed_rows (np.ndarray[int]): Entity indices with changes
            - profile_dict (Dict): Sub-phase timings if GOSSIP_PROFILE=1
    """
    N = len(adapter._entities)

    # Detect value type (position vs scalar) - determines SoA structure
    value_type = token_def.get('value_type', 'float')
    is_position = (value_type == 'position')

    # Detect merge algorithm (version_based vs most_recent)
    merge_algo = token_def.get('merge', {}).get('algorithm', 'version_based')
    is_most_recent = (merge_algo == 'most_recent')

    # Extract token state to arrays (VERSION, LAST_TICK, RELIABILITY)
    # For scalar kinds: also extract VAL (float64 SoA)
    # For position kinds: skip VAL (copy 3-tuples on writeback)
    # SoA cache: reuse arrays on cache hit, skip per-entity Python loop
    with timer.time('extract'):
        # Phase 2: Per-kind cache key (adapter_id, build_seq, kind)
        adapter_id = id(adapter)
        cache_key = (adapter_id, adapter._build_seq, kind)
        state_cache = _TOKEN_STATE_CACHE.get(cache_key)

        # Check cache validity (build_seq in key, just check N)
        cache_valid = (state_cache is not None and state_cache['N'] == N)

        if cache_valid:
            # Cache hit: reuse arrays (steady state, no entity loop)
            has = state_cache['has']
            val = state_cache['val']  # None for position kinds
            version = state_cache['version']
            last_tick = state_cache['last_tick']
            reliability = state_cache['reliability']
        else:
            # Cache miss: rebuild arrays from entities
            has = np.zeros(N, dtype=bool)
            # Allocate val SoA only for scalar kinds
            val = None if is_position else np.zeros(N, dtype=np.float64)
            version = np.zeros(N, dtype=np.int64)
            last_tick = np.zeros(N, dtype=np.float64)
            reliability = np.zeros(N, dtype=np.float64)

            for row, entity in enumerate(adapter._entities):
                token = entity.knowledge_tokens.get(kind)
                if token is not None:
                    has[row] = True
                    # Only extract val for scalar kinds (position values stay in entity dict)
                    if not is_position:
                        val[row] = token['value']
                    # Compatibility shim: treat legacy tokens as version=0, last_tick=0.0, reliability=1.0
                    version[row] = token.get('version', 0)
                    last_tick[row] = token.get('last_tick', 0.0)
                    reliability[row] = token.get('reliability', 1.0)

            # Store in cache for next tick (per-kind)
            _TOKEN_STATE_CACHE[cache_key] = {
                'N': N,
                'has': has,
                'val': val,  # None for position kinds
                'version': version,
                'last_tick': last_tick,
                'reliability': reliability
            }

            # Cache cleanup: remove stale entries for this adapter (old build_seq)
            # This prevents unbounded growth when adapter rebuilds frequently
            stale_keys = [k for k in _TOKEN_STATE_CACHE.keys()
                          if k[0] == adapter_id and k[1] != adapter._build_seq]
            for stale_key in stale_keys:
                del _TOKEN_STATE_CACHE[stale_key]

    # Compute decay arrays (Phase 2: freshness and reliability from age)
    with timer.time('decay'):
        # Get decay params from token definition
        decay_params = token_def.get('decay', {})
        freshness_rate = decay_params.get('freshness_rate', 0.01)
        reliability_rate = decay_params.get('reliability_rate', 0.005)
        eviction_threshold = decay_params.get('eviction_threshold', 0.0)

        # Compute age and decay (vectorized, O(N))
        age = np.maximum(0, current_tick - last_tick)

        # Exponential decay: freshness = exp(-rate * age), reliability_current = baseline * exp(-rate * age)
        freshness = np.exp(-freshness_rate * age)
        reliability_current = reliability * np.exp(-reliability_rate * age)

        # Clamp to [0, 1] (numerical stability)
        freshness = np.clip(freshness, 0.0, 1.0)
        reliability_current = np.clip(reliability_current, 0.0, 1.0)

    # Vectorized push-pull merge (symmetric per undirected edge)
    with timer.time('merge'):
        # Get attenuation rate from token definition
        gossip_params = token_def.get('gossip', {})
        attenuation = gossip_params.get('attenuation', 0.0)

        # Compute merge masks based on algorithm
        if is_most_recent:
            # most_recent: version > last_tick > reliability_current (with epsilon)
            # Extract arrays for clarity (single indexing)
            ver_a = version[edge_a]
            ver_b = version[edge_b]
            t_a = last_tick[edge_a]
            t_b = last_tick[edge_b]
            r_a = reliability_current[edge_a]  # Use current decayed reliability for tie-break
            r_b = reliability_current[edge_b]
            src_has = has[edge_a]
            dst_has = has[edge_b]

            # Push: a → b (a is more recent than b)
            newer_src = (
                (ver_a > ver_b) |
                ((ver_a == ver_b) & (t_a > t_b + epsilon)) |
                ((ver_a == ver_b) & (np.abs(t_a - t_b) <= epsilon) & (r_a > r_b + epsilon))
            )
            mask_push = src_has & (~dst_has | newer_src)

            # Pull: b → a (b is more recent than a)
            newer_dst = (
                (ver_b > ver_a) |
                ((ver_b == ver_a) & (t_b > t_a + epsilon)) |
                ((ver_b == ver_a) & (np.abs(t_b - t_a) <= epsilon) & (r_b > r_a + epsilon))
            )
            mask_pull = dst_has & (~src_has | newer_dst)
        else:
            # version_based (default): higher version wins
            mask_push = has[edge_a] & (~has[edge_b] | (version[edge_a] > version[edge_b] + epsilon))
            mask_pull = has[edge_b] & (~has[edge_a] | (version[edge_b] > version[edge_a] + epsilon))

        # Count exchanges (bidirectional)
        exchanges_count = int(np.sum(mask_push) + np.sum(mask_pull))

        # Apply push transfers: a → b
        if np.any(mask_push):
            dst_idx = edge_b[mask_push]
            src_idx = edge_a[mask_push]
            has[dst_idx] = True
            # Copy value only for scalar kinds (position values copied on writeback)
            if not is_position:
                val[dst_idx] = val[src_idx]
            version[dst_idx] = version[src_idx]
            last_tick[dst_idx] = last_tick[src_idx]
            # Apply multiplicative attenuation to reliability
            reliability[dst_idx] = reliability_current[src_idx] * (1.0 - attenuation)
            reliability[dst_idx] = np.clip(reliability[dst_idx], 0.0, 1.0)

        # Apply pull transfers: b → a
        if np.any(mask_pull):
            dst_idx = edge_a[mask_pull]
            src_idx = edge_b[mask_pull]
            has[dst_idx] = True
            # Copy value only for scalar kinds (position values copied on writeback)
            if not is_position:
                val[dst_idx] = val[src_idx]
            version[dst_idx] = version[src_idx]
            last_tick[dst_idx] = last_tick[src_idx]
            # Apply multiplicative attenuation to reliability
            reliability[dst_idx] = reliability_current[src_idx] * (1.0 - attenuation)
            reliability[dst_idx] = np.clip(reliability[dst_idx], 0.0, 1.0)

    # Staleness eviction (per-kind threshold check)
    with timer.time('staleness_evict'):
        # Evict tokens where freshness < eviction_threshold (if threshold > 0)
        if eviction_threshold > 0.0:
            stale_mask = has & (freshness < eviction_threshold)
            if np.any(stale_mask):
                has[stale_mask] = False

    # Write back changed rows to entities
    with timer.time('writeback'):
        # For position kinds: preserve (dst, src) pairs from push/pull for value copy
        if is_position:
            # Collect push and pull pairs (dst, src) for position value copy
            push_pairs = []
            if np.any(mask_push):
                push_pairs = list(zip(edge_b[mask_push], edge_a[mask_push]))
            pull_pairs = []
            if np.any(mask_pull):
                pull_pairs = list(zip(edge_a[mask_pull], edge_b[mask_pull]))

            # Build mapping: dst -> src for all transfers
            src_map = {dst: src for dst, src in push_pairs + pull_pairs}

        # Collect all rows that may have changed: merges + evictions
        changed_rows_merge = np.concatenate([
            edge_b[mask_push] if np.any(mask_push) else np.array([], dtype=np.int32),
            edge_a[mask_pull] if np.any(mask_pull) else np.array([], dtype=np.int32)
        ])

        # Add rows affected by staleness eviction (if any)
        if eviction_threshold > 0.0:
            stale_rows = np.where(~has & (freshness < eviction_threshold))[0].astype(np.int32)
            changed_rows = np.unique(np.concatenate([changed_rows_merge, stale_rows]))
        else:
            changed_rows = np.unique(changed_rows_merge) if len(changed_rows_merge) > 0 else np.array([], dtype=np.int32)

        for row in changed_rows:
            if has[row]:
                # Token exists: write back
                if is_position:
                    # Position kind: copy 3-tuple from source entity
                    src_row = src_map.get(int(row))
                    if src_row is not None:
                        src_token = adapter._entities[src_row].knowledge_tokens.get(kind)
                        if src_token:
                            position_value = list(src_token['value'])  # Copy 3-tuple directly
                        else:
                            # Source missing token (shouldn't happen) - use [0,0,0]
                            position_value = [0.0, 0.0, 0.0]
                    else:
                        # Row not in src_map: decay-only update, keep existing position
                        existing = adapter._entities[row].knowledge_tokens.get(kind)
                        position_value = existing['value'] if existing else [0.0, 0.0, 0.0]

                    adapter._entities[row].knowledge_tokens[kind] = {
                        'kind': kind,
                        'value': position_value,
                        'version': int(version[row]),
                        'last_tick': float(last_tick[row]),
                        'reliability': float(reliability[row]),
                        'source': 'gossip'
                    }
                else:
                    # Scalar kind: use val SoA
                    adapter._entities[row].knowledge_tokens[kind] = {
                        'kind': kind,
                        'value': float(val[row]),
                        'version': int(version[row]),
                        'last_tick': float(last_tick[row]),
                        'reliability': float(reliability[row]),
                        'source': 'gossip'
                    }
            else:
                # Token evicted: remove from entity
                if kind in adapter._entities[row].knowledge_tokens:
                    del adapter._entities[row].knowledge_tokens[kind]

    # Return results
    profile_dict = timer.timings.copy() if timer.enabled else {}
    return exchanges_count, changed_rows, profile_dict


def exchange_tokens(
    entities: List[Entity],
    adapter: SpatialIndexAdapter,
    species_registry: Dict[str, any],
    current_tick: int,
    allowed_kinds: Optional[List[str]] = None,
    cap_per_entity: int = GOSSIP_EXCHANGES_PER_ENTITY  # Deprecated in v2, kept for API compat
) -> Dict:
    """
    Exchange knowledge tokens between nearby entities (Phase 2: with lifecycle).

    Algorithm v2 (2025-10-02):
    - Radius-based neighbor queries (bidirectional, per-species gossip_range_m)
    - Undirected edge set (canonical sorted tuples, deterministic ordering)
    - Push-pull hybrid protocol (symmetric exchange per edge)
    - Version + timestamp + reliability propagation with attenuation
    - float64 arrays (precision fix)

    Steps:
    1. Extract token state to arrays (has, val, version, last_tick, reliability) - float64
    2. Compute decay arrays (freshness, reliability_current) from age
    3. Radius-based neighbor queries (query_ball_point) - automatic bidirectionality
    4. Build undirected edge set (sorted tuples, deterministic)
    5. Vectorized push-pull merge with attenuation (version comparison with epsilon)
    6. Staleness eviction (per-kind threshold check)
    7. Capacity enforcement (deterministic multi-kind sort + evict if > cap)
    8. Write back changed rows

    Args:
        entities: List of Entity objects (post-movement positions)
        adapter: SpatialIndexAdapter (built with post-movement positions)
        species_registry: Dict of Species by species_id (for gossip_range_m)
        current_tick: Current simulation tick (int) for decay computation
        allowed_kinds: List of token kinds to gossip (default: GOSSIP_ALLOWED_KINDS)
        cap_per_entity: DEPRECATED - v2 uses radius-based, not k-NN cap

    Returns:
        Dict with exchanges_count, pairs_count, and optional profile timings
    """
    # Initialize profiling timer (negligible overhead when disabled)
    timer = _GossipTimer()
    timer.reset()

    if allowed_kinds is None:
        allowed_kinds = GOSSIP_ALLOWED_KINDS

    # Use pre-loaded token definitions (loaded at module import, not per-call)
    token_defs = _TOKEN_DEFINITIONS

    # Early exit if no KD-tree or no entities
    if adapter._tree is None or len(entities) == 0:
        N = len(entities) if entities else 0
        return {
            'exchanges_count': 0,
            'pairs_count': 0,
            'telemetry': {
                'degree_mean': 0.0,
                'degree_histogram': {
                    'isolated': N,
                    'sparse': 0,
                    'connected': 0
                },
                'total_entities': N
            }
        }

    N = len(adapter._entities)

    # Epsilon for version comparison (avoid floating-point equality issues)
    EPSILON = 1e-12

    # ========================================================================
    # Build undirected edge set ONCE per tick (reused across all token kinds)
    # ========================================================================
    with timer.time('edge_query'):
        # Get or compute per-row gossip ranges (cached by adapter build_seq)
        adapter_id = id(adapter)
        cache_entry = _RANGE_CACHE.get(adapter_id)

        # Check cache validity (build_seq mismatch = adapter rebuilt)
        if cache_entry is None or cache_entry['build_seq'] != adapter._build_seq:
            # Rebuild cache: compute per-row ranges from species registry
            range_by_row = np.full(N, GOSSIP_FALLBACK_RANGE_M, dtype=np.float64)
            for row, entity in enumerate(adapter._entities):
                species = species_registry.get(entity.species_id)
                if species and hasattr(species, 'gossip_range_m') and species.gossip_range_m is not None:
                    range_by_row[row] = species.gossip_range_m

            # Check if all ranges are uniform (common case: all use fallback)
            uniform = bool(np.all(range_by_row == range_by_row[0]))
            max_range = float(np.max(range_by_row))

            # Cache for next tick
            _RANGE_CACHE[adapter_id] = {
                'build_seq': adapter._build_seq,
                'range_by_row': range_by_row,
                'uniform': uniform,
                'max_range': max_range
            }
            cache_entry = _RANGE_CACHE[adapter_id]

        max_range = cache_entry['max_range']
        if max_range <= 0:
            return {
                'exchanges_count': 0,
                'pairs_count': 0,
                'telemetry': {
                    'degree_mean': 0.0,
                    'degree_histogram': {
                        'isolated': N,
                        'sparse': 0,
                        'connected': 0
                    },
                    'total_entities': N
                }
            }

        # Uniform-range fast path: single query_pairs call, no filtering
        # (Most simulations use uniform 15m fallback, so this is the hot path)
        if cache_entry['uniform']:
            all_pairs = adapter._tree.query_pairs(r=max_range, output_type='ndarray')
        else:
            # Heterogeneous ranges: query max, then filter by min(range[i], range[j])
            # (Deferred until species actually have varying ranges)
            all_pairs = adapter._tree.query_pairs(r=max_range, output_type='ndarray')
            # TODO: Vectorized distance filter when heterogeneous ranges needed

        if len(all_pairs) == 0:
            # No pairs to exchange, but still need to run lifecycle (decay + capacity enforcement)
            edge_a = np.array([], dtype=np.int32)
            edge_b = np.array([], dtype=np.int32)
            entity_degrees = np.zeros(N, dtype=np.int32)
            pairs_count = 0
        else:
            # Extract edges (already canonical: a < b, already unique)
            edge_a = all_pairs[:, 0].astype(np.int32)
            edge_b = all_pairs[:, 1].astype(np.int32)

            # Diagnostic: track entity degrees (simple bincount, no np.unique needed)
            entity_degrees = np.bincount(edge_a, minlength=N) + np.bincount(edge_b, minlength=N)

            pairs_count = len(edge_a)

    # ========================================================================
    # Process each token kind (Phase 2: multi-kind ready)
    # ========================================================================
    total_exchanges = 0

    # Sort for deterministic ordering across runs
    for kind in sorted(allowed_kinds):
        # Get token definition for this kind
        token_def = token_defs.get(kind)
        if token_def is None:
            continue  # Skip undefined kinds

        # Process this kind (extract → decay → merge → evict → writeback)
        exchanges_count_kind, changed_rows_kind, profile_kind = _process_kind(
            kind=kind,
            token_def=token_def,
            entities=entities,
            adapter=adapter,
            edge_a=edge_a,
            edge_b=edge_b,
            entity_degrees=entity_degrees,
            current_tick=current_tick,
            epsilon=EPSILON,
            timer=timer
        )

        total_exchanges += exchanges_count_kind

    # ========================================================================
    # Capacity enforcement (after all kinds processed)
    # ========================================================================
    from aquarium.constants import GOSSIP_TOKEN_CAP_DEFAULT

    # Optimization: Skip capacity enforcement if provably unnecessary
    # Each entity can hold at most one token per kind from gossip
    # If min_cap >= K (number of gossiped kinds) AND no entity has excess tokens, skip
    K = len(allowed_kinds)

    # Quick check: does any entity exceed cap?
    max_tokens_present = max((len(e.knowledge_tokens) for e in adapter._entities), default=0)

    # Find minimum capacity across all species present in this tick
    species_present = set(e.species_id for e in adapter._entities)
    min_cap = GOSSIP_TOKEN_CAP_DEFAULT  # Start with default
    for species_id in species_present:
        spec = species_registry.get(species_id)
        if spec:
            # Try attribute access (namedtuple/dataclass)
            cap = getattr(spec, 'knowledge_capacity', None) or getattr(spec, 'capacity', None)
            # Try dict access if attributes not found
            if cap is None and isinstance(spec, dict):
                cap = spec.get('knowledge_capacity') or spec.get('capacity')
            # Use default if still None
            if cap is None:
                cap = GOSSIP_TOKEN_CAP_DEFAULT
        else:
            cap = GOSSIP_TOKEN_CAP_DEFAULT
        min_cap = min(min_cap, cap)

    # Run capacity enforcement only if needed (some entity exceeds min_cap)
    if max_tokens_present > min_cap:
        with timer.time('capacity_evict'):
            # Per-entity capacity enforcement across all token kinds
            for row in range(N):
                entity = adapter._entities[row]

                # Read capacity from species_registry with robust fallback
                spec = species_registry.get(entity.species_id)
                if spec:
                    # Try attribute access (namedtuple/dataclass)
                    cap = getattr(spec, 'knowledge_capacity', None) or getattr(spec, 'capacity', None)
                    # Try dict access if attributes not found
                    if cap is None and isinstance(spec, dict):
                        cap = spec.get('knowledge_capacity') or spec.get('capacity')
                    # Final fallback to default
                    if cap is None:
                        cap = GOSSIP_TOKEN_CAP_DEFAULT
                else:
                    cap = GOSSIP_TOKEN_CAP_DEFAULT

                token_count = len(entity.knowledge_tokens)

                if token_count > cap:
                    # Collect all tokens with their decay-derived values for sorting
                    tokens_list = []
                    for tk_kind, token in entity.knowledge_tokens.items():
                        # Compute current freshness and reliability for this token
                        tk_def = token_defs.get(tk_kind, {})
                        tk_decay = tk_def.get('decay', {})
                        tk_fresh_rate = tk_decay.get('freshness_rate', 0.01)
                        tk_rel_rate = tk_decay.get('reliability_rate', 0.005)

                        tk_age = max(0, current_tick - token.get('last_tick', 0.0))
                        tk_freshness = np.exp(-tk_fresh_rate * tk_age)
                        tk_reliability_baseline = token.get('reliability', 1.0)
                        tk_reliability_current = tk_reliability_baseline * np.exp(-tk_rel_rate * tk_age)

                        tokens_list.append({
                            'kind': tk_kind,
                            'freshness': tk_freshness,
                            'reliability': tk_reliability_current,
                            'last_tick': token.get('last_tick', 0.0),
                            'token': token
                        })

                    # Sort by (freshness ↑, reliability ↑, last_tick ↑, kind_lexical)
                    tokens_list.sort(key=lambda t: (t['freshness'], t['reliability'], t['last_tick'], t['kind']))

                    # Evict first (count - cap) tokens (stalest first)
                    num_to_evict = token_count - cap
                    for i in range(num_to_evict):
                        evict_kind = tokens_list[i]['kind']
                        del entity.knowledge_tokens[evict_kind]

    # ========================================================================
    # Diagnostics and return
    # ========================================================================
    low_degree_mask = entity_degrees < MIN_GOSSIP_NEIGHBORS
    low_degree_count = np.sum(low_degree_mask)

    # Compute degree histogram telemetry (Phase 7: network health EKG)
    # Fast O(E) computation using entity_degrees already computed
    degree_mean = float(entity_degrees.mean()) if N > 0 else 0.0
    hist_isolated = int((entity_degrees == 0).sum())
    hist_sparse = int(((entity_degrees >= 1) & (entity_degrees <= 2)).sum())
    hist_connected = int((entity_degrees >= 3).sum())

    # Build result dict with optional profiling data
    result = {
        'exchanges_count': total_exchanges,
        'pairs_count': pairs_count,
        'low_degree_count': int(low_degree_count),
        'avg_degree': degree_mean,
        'telemetry': {
            'degree_mean': degree_mean,
            'degree_histogram': {
                'isolated': hist_isolated,
                'sparse': hist_sparse,
                'connected': hist_connected
            },
            'total_entities': N
        }
    }

    # Add profiling breakdown if enabled
    if timer.enabled and timer.timings:
        result['profile'] = timer.timings.copy()

    return result
