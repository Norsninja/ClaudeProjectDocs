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
        self.timings[name] = elapsed_ns / 1_000_000  # Convert to ms

    def reset(self):
        """Clear timings (called at start of each exchange_tokens call)."""
        self.timings.clear()


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
        return {'exchanges_count': 0, 'pairs_count': 0}

    N = len(adapter._entities)

    # Phase 1: Only ship_sentiment token (single kind optimization)
    kind = 'ship_sentiment'

    # Epsilon for version comparison (avoid floating-point equality issues)
    EPSILON = 1e-12

    # Extract token state to arrays (FLOAT64, VERSION, LAST_TICK)
    # SoA cache: reuse arrays on cache hit, skip per-entity Python loop
    with timer.time('extract'):
        adapter_id = id(adapter)
        state_cache = _TOKEN_STATE_CACHE.get(adapter_id)

        # Check cache validity (build_seq mismatch or N changed = adapter rebuilt)
        cache_valid = (
            state_cache is not None
            and state_cache['build_seq'] == adapter._build_seq
            and state_cache['N'] == N
        )

        if cache_valid:
            # Cache hit: reuse arrays (steady state, no entity loop)
            has = state_cache['has']
            val = state_cache['val']
            version = state_cache['version']
            last_tick = state_cache['last_tick']
            reliability = state_cache['reliability']
        else:
            # Cache miss: rebuild arrays from entities
            has = np.zeros(N, dtype=bool)
            val = np.zeros(N, dtype=np.float64)
            version = np.zeros(N, dtype=np.int64)
            last_tick = np.zeros(N, dtype=np.float64)
            reliability = np.zeros(N, dtype=np.float64)

            for row, entity in enumerate(adapter._entities):
                token = entity.knowledge_tokens.get(kind)
                if token is not None:
                    has[row] = True
                    val[row] = token['value']
                    # Compatibility shim: treat legacy tokens as version=0, last_tick=0.0, reliability=1.0
                    version[row] = token.get('version', 0)
                    last_tick[row] = token.get('last_tick', 0.0)
                    reliability[row] = token.get('reliability', 1.0)  # Phase 2: backward compat

            # Store in cache for next tick
            _TOKEN_STATE_CACHE[adapter_id] = {
                'build_seq': adapter._build_seq,
                'N': N,
                'has': has,
                'val': val,
                'version': version,
                'last_tick': last_tick,
                'reliability': reliability
            }

        # Get positions (not cached, adapter already has efficient access)
        positions = np.array([e.position for e in adapter._entities], dtype=np.float64)

    # Compute decay arrays (Phase 2: freshness and reliability from age)
    with timer.time('decay'):
        # Get token definition for decay rates
        token_def = token_defs.get(kind, {})
        decay_params = token_def.get('decay', {})
        freshness_rate = decay_params.get('freshness_rate', 0.01)  # Default 1% per tick
        reliability_rate = decay_params.get('reliability_rate', 0.005)  # Default 0.5% per tick
        eviction_threshold = decay_params.get('eviction_threshold', 0.0)

        # Compute age and decay (vectorized, O(N))
        # age[i] = current_tick - last_tick[i]
        age = np.maximum(0, current_tick - last_tick)  # Clamp negative ages to 0 (shouldn't happen)

        # Exponential decay: freshness = exp(-rate * age), reliability_current = baseline * exp(-rate * age)
        freshness = np.exp(-freshness_rate * age)
        reliability_current = reliability * np.exp(-reliability_rate * age)

        # Clamp to [0, 1] (should be unnecessary with exp, but ensures numerical stability)
        freshness = np.clip(freshness, 0.0, 1.0)
        reliability_current = np.clip(reliability_current, 0.0, 1.0)

    # Build undirected edge set using radius-based neighbor queries
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
            return {'exchanges_count': 0, 'pairs_count': 0}

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

    # Vectorized push-pull merge (symmetric per undirected edge)
    with timer.time('merge'):
        # Get attenuation rate from token definition (Phase 2)
        gossip_params = token_def.get('gossip', {})
        attenuation = gossip_params.get('attenuation', 0.0)  # Default: no attenuation

        # Push: a → b (where a has newer version)
        mask_push = has[edge_a] & (~has[edge_b] | (version[edge_a] > version[edge_b] + EPSILON))

        # Pull: b → a (where b has newer version)
        mask_pull = has[edge_b] & (~has[edge_a] | (version[edge_b] > version[edge_a] + EPSILON))

        # Count exchanges (bidirectional)
        exchanges_count = int(np.sum(mask_push) + np.sum(mask_pull))

        # Apply push transfers: a → b
        if np.any(mask_push):
            dst_idx = edge_b[mask_push]
            src_idx = edge_a[mask_push]
            has[dst_idx] = True
            val[dst_idx] = val[src_idx]           # Copy value UNCHANGED
            version[dst_idx] = version[src_idx]   # Copy version UNCHANGED
            last_tick[dst_idx] = last_tick[src_idx]  # Copy timestamp UNCHANGED
            # Phase 2: Apply multiplicative attenuation to reliability
            # reliability_dst = reliability_current[src] * (1 - attenuation)
            reliability[dst_idx] = reliability_current[src_idx] * (1.0 - attenuation)
            reliability[dst_idx] = np.clip(reliability[dst_idx], 0.0, 1.0)  # Clamp [0,1]

        # Apply pull transfers: b → a
        if np.any(mask_pull):
            dst_idx = edge_a[mask_pull]
            src_idx = edge_b[mask_pull]
            has[dst_idx] = True
            val[dst_idx] = val[src_idx]
            version[dst_idx] = version[src_idx]
            last_tick[dst_idx] = last_tick[src_idx]
            # Phase 2: Apply multiplicative attenuation to reliability
            reliability[dst_idx] = reliability_current[src_idx] * (1.0 - attenuation)
            reliability[dst_idx] = np.clip(reliability[dst_idx], 0.0, 1.0)

    # Staleness eviction (Phase 2: per-kind threshold check)
    with timer.time('staleness_evict'):
        # Evict tokens where freshness < eviction_threshold (if threshold > 0)
        if eviction_threshold > 0.0:
            stale_mask = has & (freshness < eviction_threshold)
            if np.any(stale_mask):
                has[stale_mask] = False
                # Mark changed rows for writeback (deletions)
                # Note: We'll collect all changes (merge + evict) in writeback section

    # Capacity enforcement (Phase 2: deterministic multi-kind sort + evict if > cap)
    with timer.time('capacity_evict'):
        from aquarium.constants import GOSSIP_TOKEN_CAP_DEFAULT
        cap = GOSSIP_TOKEN_CAP_DEFAULT  # Future: per-species override

        # Per-entity capacity enforcement across all token kinds
        # Note: Phase 2a processes only one kind at a time, so this won't trigger yet
        # (would need 16+ kinds to exceed cap=16). Full multi-kind logic for Phase 2b.

        # For each entity with tokens, check count across all kinds
        for row in range(N):
            entity = adapter._entities[row]
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

                    # Update SoA arrays if this is the current kind being processed
                    if evict_kind == kind:
                        has[row] = False

    # Write back changed rows to entities
    with timer.time('writeback'):
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

        # Capacity eviction changes are already applied directly to entity.knowledge_tokens
        # So we don't need to track them separately

        for row in changed_rows:
            if has[row]:
                # Token exists: write back (merge or update)
                adapter._entities[row].knowledge_tokens[kind] = {
                    'kind': kind,
                    'value': float(val[row]),         # Ensure Python float (float64)
                    'version': int(version[row]),     # Version counter
                    'last_tick': float(last_tick[row]),  # Timestamp
                    'reliability': float(reliability[row]),  # Phase 2: reliability baseline
                    'source': 'gossip'
                }
            else:
                # Token evicted: remove from entity (staleness eviction)
                if kind in adapter._entities[row].knowledge_tokens:
                    del adapter._entities[row].knowledge_tokens[kind]

    # Diagnostics: Log low-degree nodes (entities with < MIN_GOSSIP_NEIGHBORS)
    low_degree_mask = entity_degrees < MIN_GOSSIP_NEIGHBORS
    low_degree_count = np.sum(low_degree_mask)

    # Note: We don't log here to avoid performance impact; return diagnostics for caller to log
    # if low_degree_count > 0:
    #     logger.warning(f"{low_degree_count} entities have < {MIN_GOSSIP_NEIGHBORS} neighbors")

    # Build result dict with optional profiling data
    result = {
        'exchanges_count': exchanges_count,
        'pairs_count': pairs_count,
        'low_degree_count': int(low_degree_count),  # Diagnostic
        'avg_degree': float(np.mean(entity_degrees)) if N > 0 else 0.0  # Diagnostic
    }

    # Add profiling breakdown if enabled
    if timer.enabled and timer.timings:
        result['profile'] = timer.timings.copy()

    return result
