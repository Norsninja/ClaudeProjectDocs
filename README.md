# Shipmind Project Documentation

Section 1: Shipmind – Design Overview
Goal

Create a storytelling exploration game where the player and an onboard AI (the Shipmind) collaboratively discover truths about an alien underwater world (Titan, Enceladus, or similar). Unlike scripted quest games, neither the player nor the AI knows the full truth — both are piecing it together from data, environment, and fragments of a hidden history.

Core Premise

Player role: pilot, engineer, and survivor. Operates the ship’s systems, drives exploration, manages survival.

Shipmind role: GPT-powered AI companion. Interprets sensor data, proposes actions, narrates hypotheses, speculates, and forms a bond with the player through emergent dialogue.

World: a procedurally enriched environment seeded with developer-authored facts, relics, events, and signals. Truths are latent in the database, not pre-written quests.

Narrative Mechanics

Bond formation: player choices (whether to follow or ignore AI suggestions) shape tone, trust, and emergent submind personalities.

Amnesia mechanic: to manage context/token limits, the AI “forgets” periodically — narratively justified as memory flushes or damaged archives.

Procedural storytelling: facts/events/relics are stored in a database (Neo4j graph). The AI retrieves fragments contextually to speculate, hypothesize, and propose actions.

Data Structure

Hierarchy spine: World → Biomes → Regions → Sites.

Semantic spine: Species, Relics, Signals, Events, Mechanisms.

Relationships:

Species/Relics → EMITS → Signals

Signals → INDICATES → Mechanisms

Events → OCCURS_IN → Biomes/Regions/Sites

Species → BELONGS_TO_FAMILY → Family → Culture

AI Integration

Input: structured JSON state (ship telemetry, sensor readings, current node ID, retrieved facts).

Output: concise JSON commentary (evidence, inference, speculation, proposed actions).

Memory: short-term sliding window, plus episodic summaries tied to world nodes.

Voice: one main “Captain AI” plus optional subminds for nav, sonar, engineering, biology, security.

Section 2: Schema Developer – Design Overview
Goal

Build a lightweight schema inspector/workbench to design and validate game data hierarchies before coding. Ensures projects (like Shipmind, Time Detectives, NewsPlanetAI) have a solid, consistent schema that can export to multiple backends (Neo4j, Postgres, JSON runtime packs).

Core Premise

Hierarchy-first: everything begins as a tree (like a file manager).

Cross-links supported: schema includes relationships, but the primary UX is hierarchical.

Data-agnostic: no gameplay content at this stage, only schema scaffolding and placeholders.

Project-agnostic: usable across all Mike’s future projects.

Files

taxonomy.yaml

Defines types, allowed children, attributes, and relationship types.

Example: World → Biome → Region → Site, with attrs (slug, name, notes).

outline.yaml

A tree of planned entities (IDs, names, types).

Example: Titan world skeleton with placeholder species/signals.

links.yaml

Planned cross-links between nodes (e.g., Species emits Signal).

Validation

Parent→child legality (taxonomy-driven).

Unique IDs (namespace:type/slug).

Attribute completeness & typing.

Relationship domain/range checks.

Exports

Neo4j: Cypher DDL, node creation, HAS_CHILD, typed relationships.

Postgres: Per-type tables + bridge tables, or generic JSONB entity mode.

Runtime packs: JSON/JSONL (types.core.json, outline.core.jsonl, links.core.jsonl, indices.core.json).

Editor UX

Tree view: collapsible hierarchy, drag/drop, add/move/delete nodes.

Node panel: edit attributes, view/add links, notes.

Relationship matrix: inspect and plan cross-links.

Problems panel: validation warnings/errors.

Export: one-click to Cypher/SQL/JSON packs.

Portability

Global taxonomy: common patterns (World, Biome, Event).

Project taxonomies: extend/override global for specific games.

Switchable projects: same UI, different taxonomy+outline files.

Scale readiness

Virtualized tree rendering for 100k+ nodes.

Outline chunking (per-region files, merged at load).

Strict vs. Warn validation modes.

CI-ready exporters (deterministic outputs).