"""
Spatial Index Adapter API

Provides stable interface for nearest/neighbor queries and sensor cone queries.
Phase 3: O(n) fallback implementations
Phase 4: scipy.cKDTree backend (drop-in replacement, no call site changes)

See: project/docs/SPATIAL_ADAPTER_API.md for full specification
"""

import numpy as np
import time
from typing import List, Optional, Set, Tuple, Dict
from .entity import Entity
from .data_types import EntityHit, SpatialQueryResult
from .constants import (
    USE_CKDTREE,
    CKDTREE_LEAFSIZE,
    CKDTREE_WORKERS,
    USE_PER_TAG_TREES,
    USE_LEGACY_BATCH,
    ACOUSTIC_DEFAULT_AMPLITUDE,
    ACOUSTIC_DEFAULT_PEAK_HZ,
    BIOLUM_DEFAULT_INTENSITY,
    BIOLUM_DEFAULT_WAVELENGTH,
    VENT_THERMAL_BASE_DELTA
)

# Conditional import of scipy (optional dependency)
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None


# ============================================================================
# Phase 3: O(n) Fallback Implementations
# ============================================================================

def find_nearest_by_tag(
    source: Entity,
    tag: str,
    all_entities: List[Entity],
    max_distance: Optional[float] = None
) -> Optional[Entity]:
    """
    Find nearest entity with specified tag (same biome).

    Phase 3: O(n) linear search
    Phase 4: cKDTree spatial query (same signature)

    Args:
        source: Reference entity
        tag: Tag to search for (e.g., "predator", "ship")
        all_entities: All entities in same biome
        max_distance: Optional distance limit (None = unlimited)

    Returns:
        Nearest entity with tag, or None if not found/beyond max_distance

    Tie-breaking:
        Equal distances resolved by instance_id lexicographic order
    """
    nearest = None
    min_distance = float('inf')

    # Filter to same biome and matching tag
    candidates = [e for e in all_entities
                  if e.biome_id == source.biome_id
                  and e.instance_id != source.instance_id
                  and tag in e.tags]

    for entity in candidates:
        distance = np.linalg.norm(entity.position - source.position)

        # Check max_distance constraint
        if max_distance is not None and distance > max_distance:
            continue

        # Update nearest (with tie-breaking by instance_id)
        if distance < min_distance or (distance == min_distance and entity.instance_id < nearest.instance_id):
            min_distance = distance
            nearest = entity

    return nearest


def neighbors_within(
    source: Entity,
    radius: float,
    all_entities: List[Entity],
    tag_filter: Optional[Set[str]] = None
) -> List[Entity]:
    """
    Find all entities within radius (same biome).

    Phase 3: O(n) linear search with post-sort
    Phase 4: cKDTree query_ball_point (same signature)

    Args:
        source: Reference entity
        radius: Search radius in meters
        all_entities: All entities in same biome
        tag_filter: Optional tag filter (e.g., {'mobile'})

    Returns:
        List of entities within radius, sorted by distance then instance_id
        (excludes source entity)

    Use case:
        Phase 6 gossip range queries
    """
    neighbors = []

    for entity in all_entities:
        # Skip self
        if entity.instance_id == source.instance_id:
            continue

        # Skip wrong biome
        if entity.biome_id != source.biome_id:
            continue

        # Apply tag filter
        if tag_filter is not None:
            if not any(tag in entity.tags for tag in tag_filter):
                continue

        # Check distance
        distance = np.linalg.norm(entity.position - source.position)
        if distance <= radius:
            neighbors.append((distance, entity))

    # Sort by distance, then instance_id for determinism
    neighbors.sort(key=lambda pair: (pair[0], pair[1].instance_id))

    return [entity for _, entity in neighbors]


# ============================================================================
# Phase 4: SpatialIndexAdapter Class with cKDTree
# ============================================================================

class SpatialIndexAdapter:
    """
    Spatial index adapter with stable API.

    Backend selection via constants.USE_CKDTREE:
    - True: Uses scipy.cKDTree for O(log N) queries
    - False: Uses O(n) fallback implementations

    Phase 4 features:
    - build(entities) -> constructs cKDTree or stores entity list
    - find_nearest_by_tag() -> cKDTree query or O(n) scan
    - neighbors_within() -> query_ball_point or O(n) scan
    """

    def __init__(self, use_ckdtree: Optional[bool] = None, leafsize: Optional[int] = None):
        """
        Initialize spatial adapter.

        Args:
            use_ckdtree: Override USE_CKDTREE constant (for testing)
            leafsize: Override CKDTREE_LEAFSIZE constant (for testing)
        """
        self._entities: List[Entity] = []
        self._tree: Optional[cKDTree] = None
        self._use_ckdtree = use_ckdtree if use_ckdtree is not None else USE_CKDTREE
        self._leafsize = leafsize if leafsize is not None else CKDTREE_LEAFSIZE

        # Build sequence counter (incremented on every build for cache invalidation)
        self._build_seq: int = 0

        # Phase 5: Stable index mapping for batch queries
        self._row_of_id: dict = {}  # entity_id -> row index
        self._id_of_row: List[str] = []  # row index -> entity_id
        self._biome_of_row: List[str] = []  # row index -> biome_id

        # Phase 5: Tag masks for fast filtering (hot tags only)
        self._masks: dict = {}  # tag -> np.ndarray[bool] shape (N,)

        # Phase 5 refined: Per-tag, per-biome subtrees
        self._tag_biome_trees: Dict[Tuple[str, str], Tuple[cKDTree, np.ndarray, np.ndarray]] = {}
        # Key: (tag, biome_id)
        # Value: (kdtree, rows_subset[int64], instance_ids_subset[str])

        # Phase 5 refined: Build timing (for performance breakdown logging)
        self.last_build_main_ms: float = 0.0  # Time to build main tree
        self.last_build_tag_ms: float = 0.0   # Time to build all subtrees (sum)

        # Phase 6: Thermal providers for sensor queries
        self._thermal_providers: List[Dict] = []  # List of {center, influence_radius, thermal_base_delta}

        # Validate cKDTree availability
        if self._use_ckdtree and not SCIPY_AVAILABLE:
            print("[WARN] cKDTree requested but scipy not available, falling back to O(n)")
            self._use_ckdtree = False

    def build(
        self,
        entities_or_positions,
        refs: Optional[List[Entity]] = None,
        thermal_centers: Optional[np.ndarray] = None,
        thermal_base_deltas: Optional[np.ndarray] = None,
        thermal_influences: Optional[np.ndarray] = None
    ):
        """
        Build spatial index from entities.

        Fast path (Phase 4b): Pass positions array and refs list
        Legacy path: Pass entities list only

        cKDTree mode: Constructs tree from positions
        Fallback mode: Just stores entity list

        Args:
            entities_or_positions: Either List[Entity] or np.ndarray positions (Nx3)
            refs: Optional entity list (required if first arg is positions array)
            thermal_centers: Optional (M, 3) array of vent centers
            thermal_base_deltas: Optional (M,) array of base temperature deltas (°C)
            thermal_influences: Optional (M,) array of influence radii (meters)
        """
        # Store thermal providers for sensor queries (Phase C)
        if thermal_centers is not None:
            self._thermal_centers = thermal_centers
            self._thermal_base_deltas = thermal_base_deltas
            self._thermal_influences = thermal_influences
        else:
            # Empty arrays (M=0)
            self._thermal_centers = np.empty((0, 3), dtype=np.float64)
            self._thermal_base_deltas = np.empty(0, dtype=np.float64)
            self._thermal_influences = np.empty(0, dtype=np.float64)
        # Fast path: prebuilt positions array
        if isinstance(entities_or_positions, np.ndarray):
            if refs is None:
                raise ValueError("refs required when passing positions array")
            self._entities = refs
            positions = entities_or_positions
        else:
            # Legacy path: extract positions from entities
            self._entities = entities_or_positions
            if self._use_ckdtree and len(self._entities) > 0:
                positions = np.array([e.position for e in self._entities], dtype=np.float64)
            else:
                positions = None

        # Increment build sequence (for cache invalidation in consumers)
        self._build_seq += 1

        # Phase 5 refined: Reset timing
        self.last_build_main_ms = 0.0
        self.last_build_tag_ms = 0.0

        # Build cKDTree if enabled (Phase 5: time main tree build)
        if self._use_ckdtree and positions is not None and len(positions) > 0:
            build_main_start = time.perf_counter()
            self._tree = cKDTree(positions, leafsize=self._leafsize)
            self.last_build_main_ms = (time.perf_counter() - build_main_start) * 1000.0
        else:
            self._tree = None

        # Phase 5: Build stable index mappings
        n = len(self._entities)
        self._row_of_id = {}
        self._id_of_row = []
        self._biome_of_row = []

        for row, entity in enumerate(self._entities):
            entity_id = entity.instance_id  # Use instance_id as unique identifier
            self._row_of_id[entity_id] = row
            self._id_of_row.append(entity_id)
            self._biome_of_row.append(entity.biome_id)

        # Phase 5: Build tag masks for hot tags (predator, ship)
        # Only compute masks for tags we'll query in batch
        hot_tags = ['predator', 'ship']
        self._masks = {}

        for tag in hot_tags:
            mask = np.zeros(n, dtype=bool)
            for row, entity in enumerate(self._entities):
                if tag in entity.tags:
                    mask[row] = True
            self._masks[tag] = mask

        # Phase 5 refined: Build per-tag, per-biome subtrees (time subtree builds)
        self._tag_biome_trees = {}
        if self._use_ckdtree and USE_PER_TAG_TREES and positions is not None and n > 0:
            build_tag_start = time.perf_counter()
            self._build_tag_biome_subtrees(positions)
            self.last_build_tag_ms = (time.perf_counter() - build_tag_start) * 1000.0

    def _build_tag_biome_subtrees(self, positions: np.ndarray):
        """
        Build per-tag, per-biome cKDTree subtrees for vectorized batch queries.

        Per SD guidance:
        - Build only for (tag, biome) pairs where:
          1. Tag exists in biome (mask sum > 0)
          2. Biome has entities
        - Store: (kdtree, rows_subset, instance_ids_subset)

        Args:
            positions: Full position array (N, 3)
        """
        hot_tags = ['predator', 'ship']

        # Group rows by biome
        biome_rows = {}
        for row, biome_id in enumerate(self._biome_of_row):
            if biome_id not in biome_rows:
                biome_rows[biome_id] = []
            biome_rows[biome_id].append(row)

        # Build subtree for each (tag, biome) combination
        for tag in hot_tags:
            tag_mask = self._masks[tag]

            for biome_id, biome_row_list in biome_rows.items():
                biome_row_array = np.array(biome_row_list, dtype=np.int64)

                # Find rows in this biome that have this tag
                rows_with_tag_in_biome = []
                for row in biome_row_array:
                    if tag_mask[row]:
                        rows_with_tag_in_biome.append(row)

                if not rows_with_tag_in_biome:
                    # No entities with this tag in this biome, skip
                    continue

                rows_subset = np.array(rows_with_tag_in_biome, dtype=np.int64)
                pos_subset = positions[rows_subset]

                # Build kdtree for this subset
                kdtree = cKDTree(pos_subset, leafsize=self._leafsize)

                # Extract instance IDs for tie-breaking
                instance_ids_subset = np.array([self._id_of_row[row] for row in rows_subset])

                # Store subtree
                key = (tag, biome_id)
                self._tag_biome_trees[key] = (kdtree, rows_subset, instance_ids_subset)

    def find_nearest_by_tag(
        self,
        source: Entity,
        tag: str,
        max_distance: Optional[float] = None
    ) -> Optional[Entity]:
        """
        Find nearest entity with tag.

        cKDTree mode: Query tree, filter by tag
        Fallback mode: Delegate to O(n) function

        Args:
            source: Reference entity
            tag: Tag to search for
            max_distance: Optional distance limit

        Returns:
            Nearest entity with tag, or None
        """
        if self._use_ckdtree and self._tree is not None:
            return self._find_nearest_ckdtree(source, tag, max_distance)
        else:
            return find_nearest_by_tag(source, tag, self._entities, max_distance)

    def neighbors_within(
        self,
        source: Entity,
        radius: float,
        tag_filter: Optional[Set[str]] = None
    ) -> List[Entity]:
        """
        Find all entities within radius.

        cKDTree mode: query_ball_point, filter by tag
        Fallback mode: Delegate to O(n) function

        Args:
            source: Reference entity
            radius: Search radius
            tag_filter: Optional tag filter

        Returns:
            List of entities within radius, sorted by distance
        """
        if self._use_ckdtree and self._tree is not None:
            return self._neighbors_within_ckdtree(source, radius, tag_filter)
        else:
            return neighbors_within(source, radius, self._entities, tag_filter)

    # ========================================================================
    # Phase 5: Batch Query APIs
    # ========================================================================

    def nearest_by_tag_batch(
        self,
        source_rows: np.ndarray,
        tag: str,
        max_distance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch query: find nearest entity with tag for multiple sources.

        Per-biome grouping strategy:
        - Group source rows by biome
        - Run batch query per biome (avoids cross-biome filtering)
        - Fill result arrays by original row indices

        Args:
            source_rows: np.ndarray[int], shape (N,) - row indices of source entities
            tag: Tag to search for (e.g., 'predator', 'ship')
            max_distance: Optional distance limit (None = unlimited)

        Returns:
            Tuple of (indices, distances):
            - indices: np.ndarray[int], shape (N,) - row index of nearest, or -1 if none
            - distances: np.ndarray[float], shape (N,) - distance to nearest, or inf if none

        Determinism:
            Results sorted by (distance, instance_id) for tie-breaking
        """
        n_sources = len(source_rows)

        # Initialize result arrays (all no-match)
        result_indices = np.full(n_sources, -1, dtype=np.int32)
        result_distances = np.full(n_sources, np.inf, dtype=np.float64)

        # Early exit if tag has no entities
        if tag not in self._masks or not np.any(self._masks[tag]):
            return result_indices, result_distances

        # Group sources by biome for per-biome batching
        biome_groups = {}  # biome_id -> list of (original_index, row)
        for i, row in enumerate(source_rows):
            biome_id = self._biome_of_row[row]
            if biome_id not in biome_groups:
                biome_groups[biome_id] = []
            biome_groups[biome_id].append((i, row))

        # Process each biome group
        for biome_id, group in biome_groups.items():
            # Extract row indices for this biome
            biome_source_rows = np.array([row for _, row in group], dtype=np.int32)
            biome_positions = np.array([self._entities[row].position for row in biome_source_rows])

            # Run batch query for this biome
            if self._use_ckdtree and self._tree is not None:
                # Phase 5 refined: Route to legacy or vectorized implementation
                if USE_LEGACY_BATCH:
                    biome_indices, biome_distances = self._nearest_by_tag_batch_ckdtree_legacy(
                        biome_positions, biome_source_rows, tag, biome_id, max_distance
                    )
                else:
                    biome_indices, biome_distances = self._nearest_by_tag_batch_ckdtree(
                        biome_positions, biome_source_rows, tag, biome_id, max_distance
                    )
            else:
                biome_indices, biome_distances = self._nearest_by_tag_batch_fallback(
                    biome_positions, biome_source_rows, tag, biome_id, max_distance
                )

            # Fill result arrays by original indices
            for j, (original_i, _) in enumerate(group):
                result_indices[original_i] = biome_indices[j]
                result_distances[original_i] = biome_distances[j]

        return result_indices, result_distances

    # ========================================================================
    # cKDTree Implementation Helpers
    # ========================================================================

    def _find_nearest_ckdtree(
        self,
        source: Entity,
        tag: str,
        max_distance: Optional[float]
    ) -> Optional[Entity]:
        """
        cKDTree-based nearest search with tag filtering.

        Strategy (per SD guidance):
        - If max_distance: query_ball_point, filter by tag/biome, return nearest
        - If no max_distance: query k-nearest iteratively (8, 16, 32...) until tag match
        """
        if max_distance is not None:
            # Use query_ball_point for bounded search
            indices = self._tree.query_ball_point(source.position, r=max_distance)

            # Filter by biome, self, and tag
            candidates = []
            for idx in indices:
                entity = self._entities[idx]

                if entity.instance_id == source.instance_id:
                    continue
                if entity.biome_id != source.biome_id:
                    continue
                if tag not in entity.tags:
                    continue

                distance = np.linalg.norm(entity.position - source.position)
                candidates.append((distance, entity))

            if not candidates:
                return None

            # Sort by distance, then instance_id for determinism
            candidates.sort(key=lambda pair: (pair[0], pair[1].instance_id))
            return candidates[0][1]

        else:
            # No max_distance: iteratively query k-nearest until tag match
            k = 8  # Start with small k
            max_k = len(self._entities)

            # Always query at least once, even if k > max_k
            while True:
                # Query k nearest neighbors (capped at max_k)
                actual_k = min(k, max_k)
                distances, indices = self._tree.query(source.position, k=actual_k)

                # Handle single result (k=1 returns scalars, not arrays)
                if actual_k == 1:
                    distances = np.array([distances])
                    indices = np.array([indices])
                else:
                    # Ensure arrays for consistent iteration
                    distances = np.atleast_1d(distances)
                    indices = np.atleast_1d(indices)

                # Build candidates with distances (tree.query already returns sorted by distance)
                candidates = []
                for dist, idx in zip(distances, indices):
                    entity = self._entities[idx]

                    if entity.instance_id == source.instance_id:
                        continue
                    if entity.biome_id != source.biome_id:
                        continue
                    if tag not in entity.tags:
                        continue

                    candidates.append((dist, entity))

                if candidates:
                    # Found at least one match
                    # Distances from tree.query are already sorted, but apply tie-breaking
                    candidates.sort(key=lambda pair: (pair[0], pair[1].instance_id))
                    return candidates[0][1]

                # No match yet - if we've queried all entities, give up
                if actual_k >= max_k:
                    break

                # Otherwise double k and try again
                k = k * 2

            # No match found in entire tree
            return None

    def _neighbors_within_ckdtree(
        self,
        source: Entity,
        radius: float,
        tag_filter: Optional[Set[str]]
    ) -> List[Entity]:
        """
        cKDTree-based radius search with tag filtering.

        Uses query_ball_point for efficient radius query.
        """
        # Query ball
        indices = self._tree.query_ball_point(source.position, r=radius)

        # Filter by biome, self, and tags
        neighbors = []
        for idx in indices:
            entity = self._entities[idx]

            # Skip self
            if entity.instance_id == source.instance_id:
                continue

            # Skip wrong biome
            if entity.biome_id != source.biome_id:
                continue

            # Apply tag filter
            if tag_filter is not None:
                if not any(tag in entity.tags for tag in tag_filter):
                    continue

            distance = np.linalg.norm(entity.position - source.position)
            neighbors.append((distance, entity))

        # Sort by distance, then instance_id
        neighbors.sort(key=lambda pair: (pair[0], pair[1].instance_id))

        return [entity for _, entity in neighbors]

    # ========================================================================
    # Phase 5: Batch Query Helpers
    # ========================================================================

    def _nearest_by_tag_batch_ckdtree(
        self,
        source_positions: np.ndarray,
        source_rows: np.ndarray,
        tag: str,
        biome_id: str,
        max_distance: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized cKDTree batch query using per-tag, per-biome subtrees.

        Phase 5 refined: Replaces Python loops with single tree.query() call.
        Per SD guidance: Use subtrees, apply vectorized masks, no concatenation.

        Strategy:
        - Lookup subtree for (tag, biome)
        - Single query: d, local_idx = subtree.query(positions, k=1, workers=-1)
        - Map local → global indices
        - Apply vectorized masks: self-exclusion, max_distance
        - Return arrays directly (no Python loops)

        Args:
            source_positions: np.ndarray shape (N, 3) - positions to query from
            source_rows: np.ndarray[int] shape (N,) - row indices of sources
            tag: Tag to search for
            biome_id: Biome ID (all sources assumed same biome)
            max_distance: Optional distance limit

        Returns:
            Tuple of (indices, distances) both shape (N,)
            - indices[i] = -1 if no match found
            - distances[i] = inf if no match found
        """
        n_sources = len(source_positions)
        result_indices = np.full(n_sources, -1, dtype=np.int32)
        result_distances = np.full(n_sources, np.inf, dtype=np.float64)

        # Lookup subtree for (tag, biome)
        subtree_key = (tag, biome_id)
        if subtree_key not in self._tag_biome_trees:
            # No subtree for this (tag, biome) - no candidates exist
            return result_indices, result_distances

        kdtree, rows_subset, instance_ids_subset = self._tag_biome_trees[subtree_key]

        if len(rows_subset) == 0:
            # Empty subtree (shouldn't happen if subtree exists, but guard)
            return result_indices, result_distances

        # Vectorized query: k=1 finds single nearest in subtree
        # workers=-1 uses all cores (safe with k=1, deterministic by distance)
        d, local_idx = kdtree.query(source_positions, k=1, workers=CKDTREE_WORKERS)

        # Map local indices to global row indices
        global_idx = rows_subset[local_idx]

        # Vectorized mask 1: Self-exclusion (entity finds itself)
        mask_self = (global_idx == source_rows)
        global_idx[mask_self] = -1
        d[mask_self] = np.inf

        # Vectorized mask 2: Max distance limit
        if max_distance is not None:
            mask_far = (d > max_distance)
            global_idx[mask_far] = -1
            d[mask_far] = np.inf

        # Write results directly (no concatenation)
        result_indices[:] = global_idx
        result_distances[:] = d

        return result_indices, result_distances

    def _nearest_by_tag_batch_ckdtree_legacy(
        self,
        source_positions: np.ndarray,
        source_rows: np.ndarray,
        tag: str,
        biome_id: str,
        max_distance: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        [LEGACY] cKDTree batch query for nearest entity with tag (single biome).

        Phase 5: Replaced by vectorized implementation using subtrees.
        Kept for A/B testing via USE_LEGACY_BATCH flag.

        Strategy:
        - Use query_ball_point for bounded search (if max_distance provided)
        - Filter candidates by tag mask and self-exclusion
        - Sort by distance, tie-break by instance_id

        Args:
            source_positions: np.ndarray shape (N, 3) - positions to query from
            source_rows: np.ndarray[int] shape (N,) - row indices of sources
            tag: Tag to search for
            biome_id: Biome ID (all sources assumed same biome)
            max_distance: Optional distance limit

        Returns:
            Tuple of (indices, distances) both shape (N,)
        """
        n_sources = len(source_positions)
        result_indices = np.full(n_sources, -1, dtype=np.int32)
        result_distances = np.full(n_sources, np.inf, dtype=np.float64)

        # Get tag mask for candidates
        tag_mask = self._masks[tag]

        # Build candidate rows: same biome + has tag
        candidate_rows = []
        for row in range(len(self._entities)):
            if self._biome_of_row[row] == biome_id and tag_mask[row]:
                candidate_rows.append(row)

        if not candidate_rows:
            return result_indices, result_distances

        candidate_rows = np.array(candidate_rows, dtype=np.int32)

        if max_distance is not None:
            # Bounded search: use query_ball_point
            # Query each source position against tree
            for i, (src_pos, src_row) in enumerate(zip(source_positions, source_rows)):
                indices = self._tree.query_ball_point(src_pos, r=max_distance)

                # Filter to candidates (tag + biome) and exclude self
                valid_candidates = []
                for idx in indices:
                    if idx == src_row:  # Self-exclusion
                        continue
                    if idx in candidate_rows:
                        dist = np.linalg.norm(self._entities[idx].position - src_pos)
                        valid_candidates.append((dist, self._entities[idx].instance_id, idx))

                if valid_candidates:
                    # Sort by distance, then instance_id for determinism
                    valid_candidates.sort(key=lambda x: (x[0], x[1]))
                    result_distances[i] = valid_candidates[0][0]
                    result_indices[i] = valid_candidates[0][2]
        else:
            # Unbounded search: find nearest among candidates
            for i, (src_pos, src_row) in enumerate(zip(source_positions, source_rows)):
                valid_candidates = []
                for candidate_row in candidate_rows:
                    if candidate_row == src_row:  # Self-exclusion
                        continue
                    entity = self._entities[candidate_row]
                    dist = np.linalg.norm(entity.position - src_pos)
                    valid_candidates.append((dist, entity.instance_id, candidate_row))

                if valid_candidates:
                    # Sort by distance, then instance_id
                    valid_candidates.sort(key=lambda x: (x[0], x[1]))
                    result_distances[i] = valid_candidates[0][0]
                    result_indices[i] = valid_candidates[0][2]

        return result_indices, result_distances

    def _nearest_by_tag_batch_fallback(
        self,
        source_positions: np.ndarray,
        source_rows: np.ndarray,
        tag: str,
        biome_id: str,
        max_distance: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        O(n) fallback batch query for nearest entity with tag.

        Uses same logic as _nearest_by_tag_batch_ckdtree but with O(n) scan.

        Args:
            source_positions: np.ndarray shape (N, 3)
            source_rows: np.ndarray[int] shape (N,)
            tag: Tag to search for
            biome_id: Biome ID
            max_distance: Optional distance limit

        Returns:
            Tuple of (indices, distances) both shape (N,)
        """
        n_sources = len(source_positions)
        result_indices = np.full(n_sources, -1, dtype=np.int32)
        result_distances = np.full(n_sources, np.inf, dtype=np.float64)

        # Build candidate list: same biome + has tag
        candidates = []
        for row, entity in enumerate(self._entities):
            if entity.biome_id == biome_id and tag in entity.tags:
                candidates.append((row, entity))

        if not candidates:
            return result_indices, result_distances

        # For each source, find nearest candidate
        for i, (src_pos, src_row) in enumerate(zip(source_positions, source_rows)):
            valid_candidates = []
            for cand_row, cand_entity in candidates:
                if cand_row == src_row:  # Self-exclusion
                    continue

                dist = np.linalg.norm(cand_entity.position - src_pos)

                # Check max_distance constraint
                if max_distance is not None and dist > max_distance:
                    continue

                valid_candidates.append((dist, cand_entity.instance_id, cand_row))

            if valid_candidates:
                # Sort by distance, then instance_id
                valid_candidates.sort(key=lambda x: (x[0], x[1]))
                result_distances[i] = valid_candidates[0][0]
                result_indices[i] = valid_candidates[0][2]

        return result_indices, result_distances

    # ========================================================================
    # Phase 6: Sensor Query API
    # ========================================================================

    def _compute_optical_components(self, entity: Entity) -> dict:
        """
        Compute optical channel aggregate from entity emission sources.

        Currently returns bioluminescent only. Future: artificial, reflected.
        Sensor reading semantic: no photons (intensity==0) means no wavelength.

        Args:
            entity: Entity with base_emissions and emission_multipliers

        Returns:
            {
                'total_intensity': float (clamped [0,1]),
                'wavelength_nm': float or None (None if intensity==0),
                'components': list[str] (e.g., ['bioluminescent'])
            }
        """
        base = entity.base_emissions.get('bioluminescent', {})
        multiplier = entity.emission_multipliers.get('bioluminescent', 1.0)

        base_intensity = base.get('intensity', BIOLUM_DEFAULT_INTENSITY)
        base_wavelength = base.get('wavelength_nm', BIOLUM_DEFAULT_WAVELENGTH)

        total_intensity = np.clip(base_intensity * multiplier, 0.0, 1.0)

        # Sensor reading: no photons detected = no measurable wavelength
        wavelength_nm = base_wavelength if total_intensity > 0 else None

        return {
            'total_intensity': float(total_intensity),
            'wavelength_nm': float(wavelength_nm) if wavelength_nm is not None else None,
            'components': ['bioluminescent']
        }

    def query_cone(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        angle_deg: float,
        range_m: float,
        flags: int,
        timestamp: Optional[float] = None
    ) -> SpatialQueryResult:
        """
        Query entities within a cone sensor volume.

        Strategy:
        1. KD-tree range culling: query_ball_point(origin, range_m)
        2. Vectorized angle filtering: cos(θ) ≥ cos(angle_deg)
        3. Build EntityHit DTOs with requested fields only
        4. Order results deterministically (distance, then entity_id)

        Args:
            origin: Query origin position (3,) array
            direction: Cone direction vector (3,) - will be normalized internally
            angle_deg: Cone half-angle in degrees (e.g., 45° = 90° total cone)
            range_m: Maximum query range in meters
            flags: Bitmask of QUERY_FLAG_* constants for requested fields
            timestamp: Optional timestamp for result metadata

        Returns:
            SpatialQueryResult with entities list ordered by distance

        Edge cases:
            - Zero-length direction: returns empty result
            - Entity at origin (dist==0): included if angle covers it, distance=0
            - No entities in range: returns empty entities list
        """
        # Use current time if not provided
        if timestamp is None:
            timestamp = time.time()

        # Defensive normalization of direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            # Zero-length direction - return empty result
            return SpatialQueryResult(
                entities=[],
                timestamp=timestamp,
                query_origin=tuple(origin.tolist())
            )

        direction_normalized = direction / dir_norm

        # Step 1: KD-tree range culling (or O(n) fallback)
        if self._use_ckdtree and self._tree is not None:
            indices = self._tree.query_ball_point(origin, r=range_m)
        else:
            # Fallback: O(n) scan for entities within range
            indices = []
            for i, entity in enumerate(self._entities):
                dist = np.linalg.norm(entity.position - origin)
                if dist <= range_m:
                    indices.append(i)

        if len(indices) == 0:
            # No candidates in range
            return SpatialQueryResult(
                entities=[],
                timestamp=timestamp,
                query_origin=tuple(origin.tolist())
            )

        # Step 2: Vectorized angle filtering
        # For each candidate: v = P - origin, check dot(v_norm, dir_norm) >= cos(angle_deg)
        positions = np.array([self._entities[idx].position for idx in indices], dtype=np.float64)
        vectors = positions - origin  # (N, 3)
        distances = np.linalg.norm(vectors, axis=1)  # (N,)

        # Handle entities at origin (distance == 0)
        # Include them if angle is >= 0 (cone covers origin)
        angle_mask = np.ones(len(indices), dtype=bool)
        non_zero_mask = distances > 1e-9

        if np.any(non_zero_mask):
            # Normalize vectors where distance > 0
            vectors_normalized = np.zeros_like(vectors)
            vectors_normalized[non_zero_mask] = vectors[non_zero_mask] / distances[non_zero_mask, np.newaxis]

            # Compute cos(angle) for non-zero distances
            cos_angles = np.dot(vectors_normalized, direction_normalized)  # (N,)

            # Threshold: cos(theta) >= cos(angle_deg)
            cos_threshold = np.cos(np.radians(angle_deg))
            angle_mask[non_zero_mask] = cos_angles[non_zero_mask] >= cos_threshold

        # Entities at origin (distance ~0) are included if angle >= 0 (already True in angle_mask)

        # Apply angle mask
        filtered_indices = [indices[i] for i in range(len(indices)) if angle_mask[i]]
        filtered_distances = distances[angle_mask]

        if len(filtered_indices) == 0:
            # No entities within cone
            return SpatialQueryResult(
                entities=[],
                timestamp=timestamp,
                query_origin=tuple(origin.tolist())
            )

        # Step 3: Build EntityHit DTOs with requested fields
        hits = []
        hit_positions = []  # Collect positions for thermal computation
        for i, idx in enumerate(filtered_indices):
            entity = self._entities[idx]
            distance = filtered_distances[i]

            # Always include entity_id and species_id
            hit = EntityHit(
                entity_id=entity.instance_id,
                species_id=entity.species_id
            )

            # Populate optional fields based on flags
            if flags & QUERY_FLAG_POSITION:
                hit.pos_x = float(entity.position[0])
                hit.pos_y = float(entity.position[1])
                hit.pos_z = float(entity.position[2])

            if flags & QUERY_FLAG_VELOCITY:
                hit.vel_x = float(entity.velocity[0])
                hit.vel_y = float(entity.velocity[1])
                hit.vel_z = float(entity.velocity[2])

            if flags & QUERY_FLAG_DISTANCE:
                hit.distance = float(distance)

            # Acoustic channel
            if flags & QUERY_FLAG_ACOUSTIC:
                base = entity.base_emissions.get('acoustic', {})
                base_amp = base.get('amplitude', ACOUSTIC_DEFAULT_AMPLITUDE)
                base_hz = base.get('peak_hz', ACOUSTIC_DEFAULT_PEAK_HZ)

                multiplier = entity.emission_multipliers.get('acoustic', 1.0)
                amplitude = np.clip(base_amp * multiplier, 0.0, 1.0)

                hit.acoustic_amplitude = float(amplitude)
                # Sensor reading: no sound = no measurable frequency
                hit.acoustic_peak_hz = float(base_hz) if amplitude > 0 else None

            # Bioluminescent/Optical channels (single compute path)
            if flags & (QUERY_FLAG_BIOLUMINESCENT | QUERY_FLAG_OPTICAL):
                optical_data = self._compute_optical_components(entity)

                if flags & QUERY_FLAG_BIOLUMINESCENT:
                    hit.bioluminescent_intensity = optical_data['total_intensity']
                    hit.bioluminescent_wavelength_nm = optical_data['wavelength_nm']

                if flags & QUERY_FLAG_OPTICAL:
                    hit.optical_intensity = optical_data['total_intensity']
                    hit.optical_wavelength_nm = optical_data['wavelength_nm']
                    hit.optical_components = optical_data['components']

            # Collect position for batched thermal computation
            hit_positions.append(entity.position)

            hits.append((distance, entity.instance_id, hit))

        # Step 3b: Batched thermal computation (if THERMAL flag set)
        if flags & QUERY_FLAG_THERMAL:
            M = len(self._thermal_centers)

            if M == 0:
                # No thermal providers in biome - omit field (None → excluded from to_dict)
                for _, _, hit in hits:
                    hit.thermal_temperature_delta = None
            else:
                # Vectorized nearest-vent falloff computation
                K = len(hits)
                H = np.array(hit_positions, dtype=np.float64)  # (K, 3)
                C = self._thermal_centers  # (M, 3)

                # Compute squared distances (K, M) - avoid K×M sqrt operations
                dist_sq = ((H[:,None,:] - C[None,:,:])**2).sum(axis=2)

                # Find nearest vent per hit
                nearest_idx = np.argmin(dist_sq, axis=1)  # (K,)
                nearest_dist = np.sqrt(dist_sq[np.arange(K), nearest_idx])  # (K,) - only K sqrt ops

                # Get base_deltas and influences for nearest vents
                base = self._thermal_base_deltas[nearest_idx]  # (K,)
                infl = self._thermal_influences[nearest_idx]  # (K,)

                # Linear falloff: delta = base × (1 - distance/influence), clamped to [0, +inf)
                thermal_deltas = base * (1.0 - nearest_dist / infl)
                thermal_deltas = np.maximum(thermal_deltas, 0.0)

                # Assign to hits
                for i, (_, _, hit) in enumerate(hits):
                    hit.thermal_temperature_delta = float(thermal_deltas[i])

        # Step 4: Deterministic ordering (distance, then entity_id)
        hits.sort(key=lambda x: (x[0], x[1]))

        # Extract EntityHit objects (drop sort keys)
        ordered_hits = [hit for _, _, hit in hits]

        return SpatialQueryResult(
            entities=ordered_hits,
            timestamp=timestamp,
            query_origin=tuple(origin.tolist())
        )


# ============================================================================
# Sensor Query API (Phase 6)
# ============================================================================

# QueryFlags bitmask constants
QUERY_FLAG_POSITION = 1 << 0
QUERY_FLAG_VELOCITY = 1 << 1
QUERY_FLAG_DISTANCE = 1 << 2
QUERY_FLAG_ACOUSTIC = 1 << 3
QUERY_FLAG_THERMAL = 1 << 4
QUERY_FLAG_CHEMICAL = 1 << 5
QUERY_FLAG_MAGNETIC = 1 << 6
QUERY_FLAG_BIOLUMINESCENT = 1 << 7
QUERY_FLAG_OPTICAL = 1 << 8  # Aggregate optical (includes bioluminescent + future sources)
