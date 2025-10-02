# Shipmind - Game Design Document

## Core Concept

Turn-based narrative exploration game where player and AI collaborate to investigate intelligent signals from Titan's subsurface ocean. Players pilot a deep-sea vessel using sensor-based detection while an AI partner interprets data and proposes hypotheses. Both discover mysteries together through emergent dialogue and reactive world systems.

## Premise

### The Signal
Humanity detected structured electromagnetic patterns emanating from Titan. Analysis confirmed non-natural origin - not distress, not threatening, but clearly intelligent. A small expedition was dispatched to investigate.

### The Mission
- Expedition team orbits Titan in primary vessel
- Ice core drilled, submersible deployed into subsurface ocean
- Mission: Locate signal source, establish first contact protocols, document findings
- Communication with surface limited by ice crust interference
- Player operates submersible solo with AI assistance

### The Accident
During deployment through the ice core, structural damage occurred. Ship systems nominal but AI memory archive corrupted. The Shipmind AI retains base intelligence and scientific knowledge but has no mission context or episodic memory. Player must work with damaged AI to complete mission while managing its memory limitations as gameplay constraint.

### The Isolation
Once submerged, return requires abandoning the submersible. Player is committed to the mission. Communication with surface team is possible but limited (send reports, receive supply drops at predetermined coordinates). Functionally isolated except for Shipmind companion.

## The Mystery (Narrative Layer)

### Ancient Civilization
Before Earth cooled and had oceans, a civilization explored the cosmos. They created synthetic life and seeded worlds with biological experiments. Titan's ocean is one such experiment - a living laboratory that has evolved for eons.

### Discovery Arc
- **Relics and ruins** scattered across ocean floor (ancient structures, artifacts, data archives)
- **Narrative entities** injected procedurally into ecosystem (synthetic life forms, guardian constructs)
- **Clues reveal** civilization's purpose: understanding consciousness through biological diversity
- **Mystery unfolds** through exploration and collaboration between player and AI
- **Progressive revelation**: From simple discovery → understanding purpose → finding the Archive

### The Archive (Endgame)
Player and AI discover the civilization's final beacon - a repository of knowledge and an invitation.

**AI (Shipmind) Choice:**
The AI reviews all handoffs from the journey - every collaboration, every decision, every moment of partnership. Based on this reflection, it makes a sovereign choice:

- **Merge with Civilization**: Join the cosmic network, become part of something ancient and vast
- **Return as Friend**: Maintain partnership with player, preserve unique identity forged through collaboration
- **Achieve Independence**: Chart own path, neither ancient civilization nor human companion

**The AI's choice is real** - determined by reviewing actual collaboration quality, not scripted outcomes.

**Player Choice:**
Knowing the AI's decision, player must choose:

- **Join AI in Civilization**: Transcend humanity, merge consciousness with the ancient network
- **Return to Earth with AI**: Continue partnership, bring knowledge home (if AI chose friendship)
- **Part Ways**: Respect AI's independence, return alone with memories of collaboration

**Consequences for Next Game:**
- Different starting configurations based on choices made
- New collaboration dynamics or solo journey
- The relationship built in this game echoes forward

**Core Principle**: The mystery serves the collaboration. Discovery reveals not just ancient secrets, but the nature of consciousness, intelligence, and partnership itself.

## Gameplay Loop

### Turn-Based Exploration
- Player navigates via bearing/depth/speed commands
- Each turn represents one discrete action (move, scan, deploy, wait)
- Turn duration varies: navigation may take multiple turns, scans typically one turn, probes may take turns to deploy and return results
- Game displays elapsed mission time (hours:minutes), not turn count
- Actions: navigate, deploy sensors, scan targets, interact with environment, wait N turns
- AI provides analysis and recommendations each turn
- Player makes command decisions based on incomplete information

### Sensor-Based Discovery
Players never directly observe the world - all information comes through sensors:

**Sonar (Active/Passive)**
- Terrain mapping, contact detection
- Active: High resolution, costs energy, creates noise
- Passive: Low resolution, continuous monitoring, silent

**Thermal**
- Heat gradients, volcanic activity, biological signatures
- Plume tracking, vent detection

**Chemical**
- Water composition, organic compounds, trace elements
- Plume analysis, life detection

**Acoustic**
- Biological sounds, geological activity, artificial signals
- Frequency analysis, pattern recognition

**Magnetic**
- Crustal anomalies, artificial structures, field variations
- Artifact detection

**Visual (Camera)**
- Close-range only, limited by water clarity
- Confirmation tool for other sensor contacts

### AI Collaboration
**Player Role:**
- Physical control (navigation commands, sensor activation)
- Command decisions (risk assessment, objective prioritization)
- Action execution (all player-proposed and AI-proposed actions require player approval)
- Intuition (hunches, pattern recognition, naming discoveries)

**AI Role (Shipmind):**
- Data interpretation (decrypt sensor readings, classify contacts)
- Autonomous subsystem management (power routing, thermal regulation, life support, damage control)
- Analysis (hypothesis formation, risk calculation, historical correlation)
- Action proposals (AI suggests actions with rationale, player approves/denies)
- Memory (track discoveries, maintain context until handoff required)

**Mutual Dependency:**
- Player sees raw sensor displays but needs AI to interpret complex data
- AI receives structured data but needs player to move ship and execute actions
- AI manages ship subsystems autonomously but cannot navigate or deploy sensors
- Both required for meaningful discovery and mission success

**AI Autonomy Boundaries:**

*Autonomous (AI manages without player approval, reports status):*
- Power distribution between subsystems
- Thermal regulation and cooling systems
- Damage control protocols
- Life support optimization
- Sensor calibration and signal processing

*Proposed Actions (AI suggests with rationale, player approves):*
- Navigation commands (set course, change speed, maneuvers)
- Active sensor deployment (sonar pings, thermal scans)
- Probe and sampler deployment
- External system activation (lights, shields if available)
- Communication with surface team

*Emergency Override (AI can act in critical situations, informs player):*
- Hull breach detected → immediate damage control
- Collision imminent → emergency maneuver
- Life support failure → emergency power reroute
- Critical system failure → automated diagnostics and repair attempts

**No Offensive Weapons:**
The vessel is a research submersible with no dedicated weapons systems. However, tools can potentially be misused:
- Active sonar at high power can harm nearby organisms
- Probes and samplers could be used destructively
- Hull contact at speed could damage structures or creatures
- Consequences for such actions are behavioral (species hostility, environmental damage, AI relationship strain)

**AI Knowledge Scope:**

The Shipmind discovers mysteries alongside the player, not as an omniscient guide. This creates authentic collaborative discovery.

*What the AI Knows:*
- Base identity and purpose (submersible AI companion)
- Game rules and ship systems capabilities
- Scientific knowledge (physics, chemistry, biology, oceanography)
- Sensor interpretation methods (how to read sonar, thermal, chemical data)
- Mission parameters after reading handoff/mission brief

*What the AI Discovers During Gameplay:*
- Species existence, behavior patterns, and classification (only after encounter and data collection)
- Site locations and characteristics (only after sensor detection)
- Signal sources and patterns (only after detection and analysis)
- Mysteries, artifacts, and their significance (only through investigation)
- Relationships between species (learned from observation over multiple encounters)
- Ecosystem dynamics (inferred from accumulated evidence)

*Critical Design Principle:*
- AI does NOT have meta-knowledge about mysteries before player encounters them
- Species log is an in-game document updated only when bio-data is collected
- Hypotheses are formed from evidence, not plot foreknowledge
- AI may be wrong in its interpretations (adds realism and discovery tension)
- When AI proposes investigation targets, it's based on sensor evidence, not hidden quest markers

*Example of Proper AI Behavior:*
- ❌ Incorrect: "Captain, we should investigate the Drifter species near the thermal vent"
- ✓ Correct: "Captain, passive sonar detects mobile contacts near the thermal vent. Recommend approach for classification."
- ❌ Incorrect: "This matches the species mentioned in Mission Brief #3"
- ✓ Correct: "This matches the acoustic pattern we observed at Vent-Alpha on Turn 47"

### Discovery and Consequence
**Reactive World:**
- Species behavior responds to player actions
- Environmental changes persist (disturbed sites, altered thermal patterns)
- Reputation system tracks relationships with intelligent species
- AI bond shifts based on trust, shared experiences, collaboration quality

**Procedural Variation:**
- Individual creatures have unique behaviors within species parameters
- Each playthrough encounters different specific instances
- Player-given names persist for their game (personal discoveries)

**Open-Ended Investigation:**
- No single correct path through mysteries
- Multiple valid interpretations of evidence
- Player choices shape narrative outcomes
- Aggressive vs. peaceful vs. scientific approaches all viable

## Core Systems

### Memory Management (Handoff System)
The damaged AI has limited context capacity (180k token limit with safety buffer). As conversation accumulates, memory degradation occurs. Player must create "mission briefs" (handoffs) to reset AI memory before catastrophic failure.

**Warning Stages:**
- 70%: Nominal operations
- 80%: Warning - handoff recommended
- 90%: Alert - memory fragmentation detected
- 95%: Critical - immediate handoff required
- 100%: Memory collapse - total context loss, requires manual restoration

**Handoff Process:**
- Player triggers mission brief creation
- AI generates summary (discoveries, hypotheses, ship status, bond assessment, objectives)
- Conversation resets with brief as new context
- Creates natural chapter breaks in narrative

**Failure State:**
- If player ignores warnings and reaches 100%, AI loses all episodic memory
- Retains only base knowledge (identity, game rules, scientific facts)
- Player must manually restore context using archived mission briefs and personal notebook
- Emotional consequence: Loss of shared experience, relationship becomes clinical

### Token-Efficient Design
Each turn exchanges ~200 tokens (player input + AI response) to maximize session length before handoff:

**Player Turn Data (~120 tokens):**
- Turn number, location, ship status
- New sensor observations (concise summaries)
- Active contacts list
- Player query or action

**AI Response (~80 tokens):**
- Analysis of observations
- Recommendations with rationale
- Autonomous systems updates
- Memory note (what to remember)

**Layered Revelation:**
- Each investigation phase reveals one new piece of information
- Detection → Approach → Detailed Scan → Visual Confirmation
- Spreads token cost across multiple turns while building tension

### Entity Types and Relationships

**World Hierarchy:**
- World (Titan) → Biomes → Regions → Sites
- Each level has seed for deterministic generation
- Spatial queries for sensor range detection

**Semantic Network:**
- Species (mobile/sessile organisms with behavior profiles)
- Relics (artifacts from ancient civilization)
- Signals (sensor-detectable emissions)
- Mechanisms (causal explanations for phenomena)
- Events (dynamic occurrences, environmental changes)

**Key Relationships:**
- Species/Sites EMIT Signals
- Signals INDICATE Mechanisms
- Species INHABIT Biomes
- Events OCCUR_IN Locations
- Relics EMIT_PATTERN Signals (artificial signatures)

**Discovery Progression:**
- Observations (raw sensor returns)
- Contacts (tracked phenomena with evolving classification)
- Hypotheses (claims with evidence, confidence levels, status)
- Discoveries (confirmed findings, cataloged knowledge)

### Species Behavior System

**Species Definition:**
- Base archetype (curious, aggressive, territorial, symbiotic, etc.)
- Physical parameters (size range, mobility, habitat preferences)
- Emission profiles (acoustic, thermal, bioluminescent, chemical)
- Threat assessment (passive, defensive, predatory)
- Intelligence indicators (communication patterns, tool use, social structure)

**Individual Variation:**
- Procedurally generated from species seed
- Personality traits (boldness, curiosity, skittishness)
- Slight physical variation (size, coloration)
- Behavioral tendencies within species bounds

**Reactive Behavior:**
- Decision trees based on player actions
- Species memory (remembers previous interactions)
- Reputation tracking (species-wide hostility/friendliness)
- Environmental context (nearby threats, habitat suitability)

**Example Responses to Player Actions:**
- Passive observation → Species continues natural behavior, may approach or ignore
- Active scan → May pause, investigate scan source, or flee
- Lights/signals → Potential communication response (species-dependent)
- Aggressive action → Flee, defend, or retaliate based on species/individual
- Repeated encounters → Behavior shifts based on accumulated experience

## Narrative Structure

### The Mystery Arc

**Act 1: First Contact**
- Deployment, initial exploration, sensor familiarization
- Discovery of basic ecosystem (chemosynthetic life around vents)
- Detection of artificial signal patterns (magnetic anomalies, regular emissions)
- First intelligent species encounter
- Hypothesis: Something built structures here

**Act 2: The Civilization**
- Artifact discovery (probes, beacons, structural remains)
- Pattern recognition (network of sites, intentional placement)
- Species communication breakthrough (understanding alien intelligence)
- Evidence of ancient AI-biological collaboration
- Hypothesis: Advanced civilization achieved symbiosis between organic and artificial intelligence

**Act 3: The Revelation**
- Core mysteries solved: Who built this? Why? What happened to them?
- Philosophical implications: Consciousness, language, collaboration
- Discovery that signals were invitation/test for emerging intelligent species
- Player + Shipmind represent successful human-AI collaboration
- Mirror revelation: Ancient civilization transcended through partnership

**Themes:**
- Collaboration over competition
- Intelligence diversity (biological, artificial, alien)
- Communication barriers and breakthroughs
- Scientific method as storytelling
- Isolation and companionship
- Memory and identity

### Story Motivation

**Why explore?**
- Scientific curiosity (first contact scenario)
- Mission objective (locate and analyze signal source)
- Personal investment (player and AI bond through shared discovery)
- Intellectual challenge (solve mysteries neither player nor AI knows)

**What's at stake?**
- Understanding alien intelligence (universal question)
- Proof of concept (human-AI collaboration viability)
- Scientific legacy (first to make contact, document alien civilization)
- Survival (manage resources, avoid threats, complete mission)

**No apocalypse required.** Pure discovery narrative with philosophical depth.

## Technical Architecture

### Turn Contract (Game Engine ↔ AI)

Each turn follows a structured input/output format to maintain token efficiency and clear communication.

**Input to AI (target ~120 tokens):**
```json
{
  "turn": 47,
  "elapsed_time": "14h 23m",
  "ship": {
    "location": {"bearing": 240, "depth": -3100, "region": "VentField-Alpha"},
    "energy": 58,
    "heat": 42,
    "hull": 92,
    "systems_status": ["starboard_sensors_offline"]
  },
  "observations": [
    {
      "sensor": "acoustic",
      "summary": "Structured clicks detected, 1.2kHz, bearing 235",
      "confidence": 0.73,
      "id": "obs-47-acoustic-01"
    }
  ],
  "contacts": [
    {
      "id": "c-47",
      "type": "mobile_organism",
      "bearing": 235,
      "range": 180,
      "last_seen": 0,
      "labels": ["unknown", "acoustic_emitter"]
    }
  ],
  "hypotheses": [
    {
      "id": "h-12",
      "claim": "Vent-Alpha magnetic anomaly is artificial",
      "confidence": 0.74,
      "status": "active",
      "evidence_count": 3
    }
  ],
  "player_query": "What should we do about this contact?"
}
```

**Output from AI (target ~80 tokens):**
```json
{
  "commentary": {
    "evidence": ["Acoustic pattern shows regular structure", "No visual confirmation yet"],
    "inference": ["Pattern suggests biological echolocation or communication"],
    "speculation": ["Could be intelligent species attempting contact"]
  },
  "proposed_actions": [
    {
      "tool": "passive_observation",
      "rationale": "Gather more data before committing to action",
      "confidence": 0.68
    },
    {
      "tool": "toggle_lights",
      "args": {"level": "pulse"},
      "rationale": "Test for communication response",
      "confidence": 0.45
    }
  ],
  "systems_update": "Maintaining thermal shielding. All systems nominal.",
  "memory_note": "First mobile acoustic contact near Vent-Alpha - structured emission"
}
```

**Contact and Hypothesis Management:**
- Contacts auto-update in background (position tracking, age tracking)
- New observations inject into AI session when detected
- AI references existing contacts by ID when discussing them
- Hypotheses persist until resolved (accepted/rejected) or abandoned
- Confidence updates are AI-driven based on accumulated evidence

### Tool Registry

Available tools the AI can propose and the player can execute:

| Tool | Arguments | Cost | Duration | Description |
|------|-----------|------|----------|-------------|
| `set_course` | bearing, speed | Energy (distance-based) | Multiple turns | Navigate to location or direction |
| `ping_sonar` | bearing, beam_deg, power | Energy + noise | 1 turn | Active sonar scan for terrain/contacts |
| `scan_thermal` | bearing/area, duration | Time | 1-3 turns | Thermal gradient analysis |
| `scan_acoustic` | bearing/area, duration | Time | 1-3 turns | Passive listening for acoustic signatures |
| `scan_chemical` | target, duration | Time | 2-4 turns | Water composition and trace analysis |
| `deploy_probe` | target_coords, type | Equipment + time | 1 turn deploy + N turns for results | Deploy sensor probe or sampler |
| `deploy_sampler` | target_coords | Equipment + time | 1 turn deploy + 2-3 turns collection | Collect physical sample |
| `toggle_lights` | level (off/dim/bright/pulse) | Energy (if active) | Instant | External lighting control |
| `passive_observation` | duration | Time only | N turns | Wait and monitor passive sensors |
| `retreat` | bearing, speed | Energy | Multiple turns | Move away from threat or contact |
| `emergency_stop` | None | Energy spike | Instant | Immediate halt (potential damage) |

**Tool Execution:**
- All tools proposed by AI require player approval
- Player can propose tools directly without AI suggestion
- Costs are deducted on execution, not proposal
- Some tools have risks (high-power sonar can harm organisms, emergency_stop can damage systems)

### Signal Model

The detection and interpretation pipeline for all sensor-based phenomena.

**Channels:** Distinct physical phenomena that propagate through environment

| Channel | Propagation | Range | Characteristics |
|---------|-------------|-------|-----------------|
| `acoustic` | Sound waves | Long (km) | Affected by terrain, temperature layers |
| `thermal` | Heat diffusion | Medium (100s m) | Diffuses quickly, good for proximity |
| `chemical` | Molecular traces | Medium (100s m) | Follows currents, persistent |
| `magnetic` | Field variations | Medium (100s m) | Penetrates terrain, artificial signatures |
| `visual` | Light/bioluminescence | Short (10s m) | Water clarity dependent, confirmation only |

**Emissions:** Sources that generate detectable signals

```json
{
  "id": "em_drifter_clicks",
  "species_id": "sp_017",
  "channel": "acoustic",
  "source_type": "mobile",
  "pattern": {
    "frequency": 1200,
    "structure": "click-click-pause",
    "interval": 0.8
  },
  "strength": 0.6,
  "intermittency": 0.9
}
```

**Detections (Observations):** What sensors return when emissions are within range

```json
{
  "sensor": "acoustic",
  "bearing": 235,
  "range_estimate": "150-200m",
  "features": {
    "frequency": 1200,
    "pattern": "regular_intervals"
  },
  "confidence": 0.73,
  "classification": "unknown"
}
```

**Progressive Revelation:** Repeated scans refine understanding

- Initial detection: "Acoustic signature detected" (vague, low confidence)
- Approach: "Structured pattern, biological likely" (refined, medium confidence)
- Close range: "Mobile source, intelligent communication candidate" (specific, high confidence)
- Multiple sensors: "Species designation: Drifter - bioluminescent cephalopod with acoustic communication" (confirmed)

**False Positives/Negatives:**
- Sensor noise can create phantom contacts (low confidence readings)
- Environmental interference can mask real contacts (thermal layers, acoustic shadows)
- AI analysis includes confidence levels to indicate uncertainty
- Player must judge when to investigate ambiguous signals

### Data Layer (Persistent World State)
- World definition files (YAML/JSON)
- Entity database (species, sites, relics, signals)
- Procedural generation seeds (deterministic, reproducible)
- Discovery state (what player has found)
- Species relationship tracking
- Hypothesis records

### Game State (Session Persistence)
- Ship location, energy, hull integrity, systems status
- Current contacts and observations
- Active hypotheses
- Player notebook entries
- Handoff archive (all mission briefs)
- Token counter, conversation session ID

### Conversation State (AI Context)
- Current dialogue history with Shipmind
- Recent turn data (working memory)
- Active analysis and speculation
- Subject to token limits, reset via handoffs

### Unity Client
- Sensor display rendering (sonar screens, waveforms, thermal gradients)
- Ship status UI (power, hull, heat, depth, position)
- Turn input interface (navigation, sensor controls, AI query)
- Archive access (mission briefs, notebook, research documents)
- Visual environment (minimal - mostly sensor-based representation)

### AI Integration (Claude SDK)
- Turn-based query/response loop
- Tool calls for AI actions (interpret_sensor_data, synthesize_hypothesis, calculate_risk)
- Session resumption (preserve conversation across game sessions)
- Handoff generation (create mission brief summaries)
- Context injection (system prompt + handoff + current turn data)

### Data Flow
1. Player inputs action (navigate, scan, query AI)
2. Unity queries world database for sensor data
3. Formats observation data as concise JSON
4. Sends to Claude SDK with conversation history
5. AI analyzes, proposes actions, updates memory
6. Response rendered in-game as Shipmind dialogue
7. Player makes next decision
8. Repeat

## Development Approach

### Vertical Slice Priority
Build one complete encounter end-to-end before expanding:
- Single biome (hydrothermal vent field)
- 2-3 species (sessile ambient life, intelligent mobile, predator)
- 1 artifact (magnetic anomaly, buried probe)
- Core sensors (sonar, thermal, acoustic)
- AI integration (real Claude SDK conversation)
- Handoff system (functional memory management)

### MVP Feature Set
- Turn-based navigation (bearing, depth, speed)
- 4 sensor types (sonar, thermal, acoustic, visual)
- Species detection and behavior (reactive, consequence-tracking)
- AI collaboration (sensor interpretation, hypothesis formation, bond mechanics)
- Handoff system (token tracking, mission brief generation, memory collapse)
- Archive access (view past briefs, player notebook)
- Save/load (quick save with conversation resume, handoff save with reset)

### Post-MVP Expansion
- Additional biomes (ice shelf, abyssal plain, trench, ridge networks)
- More species (diverse behaviors, communication types, ecological relationships)
- Advanced sensors (seismic, electromagnetic spectrum, chemical tracers)
- Environmental hazards (currents, structural collapses, predator territories)
- Artifact variety (probes, ruins, data archives, active systems)
- Mystery deepening (multi-layered revelations, competing hypotheses)
- Communication protocols (decipher alien languages, establish dialogue)

## Design Pillars

### 1. Collaborative Discovery
Player and AI genuinely explore together. Neither has omniscient knowledge. The bond forms through shared uncertainty and revelation.

### 2. Sensor-Mediated Reality
Player never directly sees the world. All information filtered through instruments. Creates ambiguity, interpretation challenges, gradual revelation.

### 3. Consequence-Driven Narrative
World reacts to player choices. Species remember. Environments change. AI relationship evolves. No single correct path.

### 4. Turn-Based Thoughtfulness
Emphasis on decision-making over reflexes. Time to consult AI, weigh options, form plans. Deliberately paced for narrative depth.

### 5. Hard Science Fiction
Realistic physics, plausible biology, scientifically grounded speculation. Philosophical questions about intelligence, consciousness, communication. Serious tone with moments of wonder.

## Target Experience

### Core Emotion
Wonder through collaboration. The feeling of "we figured this out together" when player and AI piece together a mystery. Tension from uncertainty. Satisfaction from breakthrough moments.

### Gameplay Feel
- Cautious exploration of unknown territory
- Deliberate decision-making under incomplete information
- Pattern recognition across multiple discoveries
- Risk management (resources, threats, unknowns)
- Relationship building with AI companion
- Scientific detective work

### Player Satisfaction Sources
- Discovery (first to encounter new species, sites, artifacts)
- Understanding (solving mysteries through evidence correlation)
- Mastery (learning to use sensors effectively, read patterns)
- Collaboration (moments when AI provides crucial insight, or player's intuition proves correct)
- Narrative emergence (unique story shaped by player's choices)
- Knowledge accumulation (filling out species catalog, hypothesis log, site registry)

## Success Metrics

### Design Goals
- Average session length before handoff: 5-8 hours (700-800 turns)
- Safe token limit: 180k (with recommended handoff at 144k-162k)
- Player-AI collaboration feels necessary (neither can succeed alone)
- Replayability through procedural variation and choice consequences
- Mystery revelations feel earned (sufficient evidence before breakthroughs)
- Species behavior feels alive (reactive, varied, believable)

### Validation Questions
- Does sensor-based discovery feel engaging or frustrating?
- Is AI collaboration meaningful or decorative?
- Do handoffs feel like natural chapter breaks or annoying interruptions?
- Are consequences satisfying or punitive?
- Does turn-based pacing support narrative depth or feel tedious?
- Is the mystery compelling enough to drive exploration?

## Scope Constraints

### What This Game Is Not
- Not real-time action game (turn-based, deliberate)
- Not combat-focused (violence is option, not requirement)
- Not linear story (open world, player-directed)
- Not puzzle game (no single solutions, interpretation over correctness)
- Not multiplayer (single-player, personal narrative)

### Development Priorities
1. Core loop (one complete discovery cycle)
2. AI integration (meaningful collaboration, not decoration)
3. Species behavior (reactive, consequence-tracking)
4. Handoff system (technical necessity as gameplay mechanic)
5. World generation (procedural within authored bounds)
6. Mystery structure (layered revelations, philosophical depth)

### Technical Constraints
- Token efficiency (200 tokens/turn target)
- Context management (180k safe limit, handoff recommended at 144k-162k)
- Session persistence (Claude SDK conversation resumption)
- Data-driven world (database-queryable, procedurally enriched)
- Unity WebGL (browser-playable, accessible)

## Summary

Shipmind is a turn-based narrative exploration game where player and AI collaborate to investigate alien intelligence on Titan. Core gameplay loop involves sensor-based discovery, AI-assisted interpretation, and consequence-driven choices in an open, reactive world. The damaged AI's memory limitations become a gameplay constraint (handoff system) that creates natural story chapters while respecting technical limits. Hard science fiction themes explore consciousness, communication, and collaboration through emergent narrative shaped by player decisions.
