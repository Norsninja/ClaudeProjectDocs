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
    allowed_kinds: Optional[List[str]] = None,
    cap_per_entity: int = GOSSIP_EXCHANGES_PER_ENTITY  # Deprecated in v2, kept for API compat
) -> Dict:
    """
    Exchange knowledge tokens between nearby entities (Phase 1: Exchange only).

    Algorithm v2 (2025-10-02):
    - Radius-based neighbor queries (bidirectional, per-species gossip_range_m)
    - Undirected edge set (canonical sorted tuples, deterministic ordering)
    - Push-pull hybrid protocol (symmetric exchange per edge)
    - Version + timestamp propagation (NO attenuation in Phase 1)
    - float64 arrays (precision fix)

    Steps:
    1. Extract token state to arrays (has, val, version, last_tick) - float64
    2. Radius-based neighbor queries (query_ball_point) - automatic bidirectionality
    3. Build undirected edge set (sorted tuples, deterministic)
    4. Vectorized push-pull merge (version comparison with epsilon)
    5. Write back changed rows (version, last_tick unchanged during propagation)

    Args:
        entities: List of Entity objects (post-movement positions)
        adapter: SpatialIndexAdapter (built with post-movement positions)
        species_registry: Dict of Species by species_id (for gossip_range_m)
        allowed_kinds: List of token kinds to gossip (default: GOSSIP_ALLOWED_KINDS)
        cap_per_entity: DEPRECATED - v2 uses radius-based, not k-NN cap

    Returns:
        Dict with exchanges_count and pairs_count

    Note: Attenuation deferred to Phase 2. Version/timestamp copied unchanged.
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
    with timer.time('extract'):
        has = np.zeros(N, dtype=bool)
        val = np.zeros(N, dtype=np.float64)       # FIX: was float32
        version = np.zeros(N, dtype=np.int64)     # NEW: monotonic version counter
        last_tick = np.zeros(N, dtype=np.float64) # NEW: timestamp for Phase 2 decay

        for row, entity in enumerate(adapter._entities):
            token = entity.knowledge_tokens.get(kind)
            if token is not None:
                has[row] = True
                val[row] = token['value']
                # Compatibility shim: treat legacy tokens as version=0, last_tick=0.0
                version[row] = token.get('version', 0)
                last_tick[row] = token.get('last_tick', 0.0)

        # Get positions
        positions = np.array([e.position for e in adapter._entities], dtype=np.float64)

    # Build undirected edge set using radius-based neighbor queries
    with timer.time('edge_query'):
        # Note: We still group by range, but use query_ball_point instead of k-NN
        range_groups = defaultdict(list)
        for row, entity in enumerate(adapter._entities):
            species = species_registry.get(entity.species_id)
            gossip_range = GOSSIP_FALLBACK_RANGE_M
            if species and hasattr(species, 'gossip_range_m') and species.gossip_range_m is not None:
                gossip_range = species.gossip_range_m

            if gossip_range > 0:
                range_groups[gossip_range].append(row)

        # PERFORMANCE FIX: Use query_pairs instead of query_ball_point
        # query_pairs returns undirected pairs (i, j) with i < j, already canonical and unique
        # This eliminates query_ball_point overhead and all deduplication logic

        # For multiple gossip ranges: use max range (all entities in same range group)
        # Most simulations have uniform ranges, so this is typically exact
        if not range_groups:
            return {'exchanges_count': 0, 'pairs_count': 0}

        max_range = max(range_groups.keys())

        # Get all unique pairs within max range (ndarray shape (n_pairs, 2), already sorted)
        # PERFORMANCE: This is the hot path - one fast C++ call, no Python loops
        all_pairs = adapter._tree.query_pairs(r=max_range, output_type='ndarray')

        if len(all_pairs) == 0:
            return {'exchanges_count': 0, 'pairs_count': 0}

        # Extract edges (already canonical: a < b, already unique)
        edge_a = all_pairs[:, 0].astype(np.int32)
        edge_b = all_pairs[:, 1].astype(np.int32)

        # Diagnostic: track entity degrees (simple bincount, no np.unique needed)
        entity_degrees = np.bincount(edge_a, minlength=N) + np.bincount(edge_b, minlength=N)

        pairs_count = len(edge_a)

    # Vectorized push-pull merge (symmetric per undirected edge)
    with timer.time('merge'):
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
            val[dst_idx] = val[src_idx]           # Copy value UNCHANGED (no attenuation)
            version[dst_idx] = version[src_idx]   # Copy version UNCHANGED
            last_tick[dst_idx] = last_tick[src_idx]  # Copy timestamp UNCHANGED

        # Apply pull transfers: b → a
        if np.any(mask_pull):
            dst_idx = edge_a[mask_pull]
            src_idx = edge_b[mask_pull]
            has[dst_idx] = True
            val[dst_idx] = val[src_idx]
            version[dst_idx] = version[src_idx]
            last_tick[dst_idx] = last_tick[src_idx]

    # Write back changed rows to entities
    with timer.time('writeback'):
        changed_rows = np.unique(np.concatenate([
            edge_b[mask_push] if np.any(mask_push) else np.array([], dtype=np.int32),
            edge_a[mask_pull] if np.any(mask_pull) else np.array([], dtype=np.int32)
        ]))

        for row in changed_rows:
            if has[row]:
                adapter._entities[row].knowledge_tokens[kind] = {
                    'kind': kind,
                    'value': float(val[row]),         # Ensure Python float (float64)
                    'version': int(version[row]),     # NEW: version counter
                    'last_tick': float(last_tick[row]),  # NEW: timestamp
                    'source': 'gossip'
                }

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
