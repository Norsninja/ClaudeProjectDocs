# Sensor Simulation in 3D Environments - External Research

**Research Date:** 2025-09-30
**Focus:** Sonar, thermal, and acoustic sensor simulation for underwater games/simulations
**Scope:** Real-world implementations, performance patterns, battle-tested solutions

---

## Executive Summary

Sensor simulation in 3D games requires careful trade-offs between physical accuracy and performance. The critical findings:

1. **Raycasting remains the foundation** but raw raycasting doesn't scale - developers report needing thousands of rays for acceptable sonar resolution, causing major performance issues. The solution is a **two-phase approach**: cheap spatial queries (OverlapSphere) followed by selective raycasting.

2. **Unity's Job System with RaycastCommand** provides 10x+ performance gains through multithreading, but introduces 1-frame latency. For real-time sensor systems with 100+ simultaneous queries, this is the only viable CPU-based approach.

3. **GPU-based solutions exist** but have significant limitations - they raycast against MeshRenderers (not physics colliders) and require expensive GPU-to-CPU reads. They're only worthwhile for specific scenarios like skinned mesh raycasting.

---

## Implementation Patterns

### 1. Vision Cone / FOV Detection (Most Battle-Tested)

**Pattern:** Two-stage detection with angle filtering
- Stage 1: `Physics.OverlapSphere` to find candidates within range
- Stage 2: Filter by angle (`Vector3.Angle` between forward and target direction)
- Stage 3: Raycast only the filtered candidates to check line-of-sight

**Code Pattern (from Unity community):**
```csharp
// Stage 1: Cheap radius check
Collider[] candidates = Physics.OverlapSphere(transform.position, detectionRadius, targetLayer);

// Stage 2: Angle filtering
foreach (Collider candidate in candidates) {
    Vector3 directionToTarget = (candidate.transform.position - transform.position).normalized;
    float angle = Vector3.Angle(transform.forward, directionToTarget);

    if (angle < fieldOfViewAngle / 2f) {
        // Stage 3: Line of sight check
        if (Physics.Raycast(transform.position, directionToTarget, out RaycastHit hit, detectionRadius)) {
            if (hit.collider == candidate) {
                // Target detected
            }
        }
    }
}
```

**Source Repositories:**
- arthurgonze/2D-Field-of-View: FOV detection for top-down stealth games (Unity 2020.3)
- ntk4/UnitySensorySystem: Modular sensor system with multiple view cones per NPC
- muveso/Unity-Detection-Sensor: Optimized raycast detection system

**Performance:** Cheap enough for dozens of NPCs per frame. One developer reports "you can get away with a lot of raycasts without bogging down FPS" as long as layer masks are used.

---

### 2. Sonar Simulation Patterns

**GPU-Based Ray Tracing (Research-Grade):**
- **Pattern:** Hybrid rasterization + ray tracing pipeline
- Rasterization computes primary intersections, ray tracing handles reflections
- Each sonar beam comprises multiple vertical computational rays
- Achieves real-time performance for multibeam sonar simulation

**Example Implementation:** RomeldaB/Sonar-Simulation (Unity 2019.4.25)
- Simulates sonar images of 3D scenes using ray tracing
- Uses fan-shaped beam arrangement (multibeam pattern)
- Each beam simplified as cone with 3dB equivalent beam angle

**Multibeam Sonar Structure (from research):**
- Fan-shaped arrangement of N identical beams
- Each beam represented as cone with beamwidth = 3dB angle
- Time series generated for beamformed directions
- Implemented in Gazebo with ROS integration for real-time sensor data

**Production Projects:**
- srmauvsoftware/URSim: UUV simulator with side-scan and multibeam sonar (ROS + Unity3D)
- lafith/AUV-Simulator-Unity: Autonomous underwater vehicle simulator
- Scowen/deep-blue: Submarine simulation game (Unity, C#/GLSL)

**Critical Finding:** Early Unity forum discussion (2010) reports raw raycasting for sonar requires "thousands of rays" for acceptable resolution, causing severe slowdowns. Modern implementations use GPU-accelerated approaches or hybrid methods.

---

### 3. Batch Raycasting with Job System

**Unity RaycastCommand Pattern:**
```csharp
// Setup
NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(rayCount, Allocator.TempJob);
NativeArray<RaycastHit> results = new NativeArray<RaycastHit>(rayCount, Allocator.TempJob);

for (int i = 0; i < rayCount; i++) {
    commands[i] = new RaycastCommand(origin, directions[i], maxDistance);
}

// Execute asynchronously
JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, batchSize);

// Do other work here...

// Get results
handle.Complete();
```

**Performance Characteristics:**
- 10x+ performance gains through multithreading (reported by community)
- One developer: "50 FPS with 100+ raycast bullets in FixedUpdate" - Job System can provide the needed boost
- 1-frame latency due to async execution
- Requires Unity.Collections (NativeArray) - zero GC allocations
- Works on CPU (not GPU), uses Job System for parallelization

**Critical Optimization:** Use `NativeArrayOptions.UninitializedMemory` when writing to entire array immediately after creation.

**Repository:** adammyhre/Unity-Batch-Raycasting

---

### 4. Spatial Partitioning for Sensor Queries

**BVH vs Octree Trade-offs:**

| Structure | Best For | Performance Characteristics |
|-----------|----------|---------------------------|
| **BVH** | Dynamic scenes, clustered objects | - More flexible and simple<br>- Objects in one volume only (tested once)<br>- Easy to update (translate bounding volume)<br>- Preferred by most engines |
| **Octree** | Static scenes, uniform distribution | - Can eliminate half of children per axis check<br>- Good for point queries<br>- Requires rebuild for dynamic scenes<br>- 512^3 voxel grid: 300ms (SVO) vs 719ms (step-through) |

**Key Insight from Game Programming Patterns:**
"Spatial partitions exist to knock an O(n) or O(n²) operation down to something more manageable when doing queries to find objects by location."

**Recommendation:** For sensor queries in dynamic underwater environments, BVH is preferred. Unity's physics engine already uses spatial partitioning internally.

**Spatial Hash for Broad Phase:**
- Place objects in 2D/3D grid cells based on position
- Query only objects in relevant cells
- Significantly faster broad-phase before narrow-phase collision/detection
- Repository: adam-arthur/spatial-hashmap

---

### 5. Acoustic Propagation Simulation

**Real-Time Game Implementations:**

**Microsoft Project Triton/Project Acoustics:**
- Physically models sound propagation given scene shape and materials
- Automatic modeling of occlusion and reverberation
- Models wave effects like diffraction in complex 3D scenes
- Production-ready for games

**GSound (UNC Chapel Hill):**
- First practical real-time system for game-like scenes
- Sound reflection and diffraction paths validated using ray-based occlusion queries
- Designed for current-gen platforms

**Unity/Unreal Integration:**
- Roblox Acoustic Simulation: 3D audio adapts to environment, sounds muffled by structures, bending and reflecting
- Unreal Engine 4 + Wwise: Examined in studies vs commercial geometrical acoustics software
- Unity PROPAGATE: Real-time sound propagation package

**Simplified Approaches for Games:**
- Occlusion: Ray-based queries from source to listener
- Absorption: Material-based attenuation coefficients
- Reflection: Simple bounce counting with energy decay
- Diffraction: Huygens principle simplification (see below)

**Digital Huygens Model (DHM):**
- Based on Huygens-Fresnel principle: each wavefront point = new spherical wave source
- Implemented on FPGA (XC5VLX330T) for real-time performance
- Sequences of impulses scattered according to Huygens' principles
- GPU implementations enable real-time results for games

**Key Trade-off:** Full wave-based simulation vs ray-based approximation
- Wave-based: Physically accurate, computationally expensive
- Ray-based: Fast approximation, good enough for games
- Hybrid: Ray-based with wave effect corrections (diffraction zones)

---

### 6. Thermal/Field Sensor Simulation

**3D Texture Lookup Pattern:**
```csharp
// Sample thermal field at world position
Vector3 normalizedPos = (worldPos - fieldOrigin) / fieldSize;
Color sample = thermalField3D.Sample(normalizedPos);
float temperature = sample.r * temperatureRange;
```

**Voxel Grid Implementation (from GameDev Stack Exchange):**
- Use double-buffering to avoid directional bias in heat propagation
- HeatData struct: { temperature, transferRate, dissipateRate }
- Update cells from neighbor values without creating artifacts
- Can represent heat diffusion maps, simplified physics

**GPU Compute Shader Approach:**
- Store thermal field in 3D texture (RenderTexture with volumetric support)
- Update field on GPU using compute shaders
- Sample via Shader Graph in materials or scripts
- Millions of voxels can be processed per frame

**Perlin Noise for Sensor Uncertainty:**
- Combine Perlin noise (speckle/natural variation) + Gaussian noise (sensor error)
- Perlin amplitude based on sensor resolution/roughness
- Gaussian amplitude based on sensor accuracy spec
- Used in laser profilometer simulations, applicable to thermal imaging

**Example Use Cases:**
- Minecraft/Terraria: Perlin noise for terrain, biomes, weather
- Fire propagation: Invisible voxel grid, each cell has HP/radius/material
- Energy2D: Interactive heat transfer simulation matching infrared thermography

**Repository:** mbaske/grid-sensor (Unity ML-Agents Grid Sensor with frustum culling)

---

## Performance Data

### Raycasting Performance

**Unity Physics.Raycast costs (from community profiling):**
- Single raycast: "Expensive but accurate"
- OverlapSphere: "Cheap" - described as ~10x+ faster than equivalent raycasts
- SphereCast: "Same expense category as Raycast"
- Layer masks: "Raycasts are faster if you use layer masks to ignore bulk of colliders"

**Non-Allocating Variants:**
- `Physics.Raycast` allocates heap memory (triggers GC)
- `Physics.RaycastNonAlloc` writes to pre-allocated array (zero GC)
- Critical for sensor systems called every frame

**Batch Performance:**
- Job System + RaycastCommand: 10x+ improvement over standard raycasts
- 100+ raycasts in FixedUpdate: ~50 FPS without batching
- Latency cost: 1 frame delay due to async execution

### GPU Culling Performance

**Frustum Culling Test (from Unity GPU culling experiments):**
- Scene: 400,000 trees
- Regular Unity: 10-12 FPS
- GPU culling: 2500 FPS
- **210x performance improvement**

**Note:** This is for rendering culling, not physics queries. Physics.Raycast runs on CPU.

### Spatial Partitioning Performance

**Octree SVO Test (512x512x512 grid, 600,000 voxels):**
- SVO method: 300ms
- Step-through: 719ms
- **2.4x improvement**

**BVH for Dynamic Scenes:**
- Translate 100,000 clustered vertices: Move single bounding volume
- Octree equivalent: Must update grid cells for all vertices
- **Orders of magnitude difference for dynamic content**

### NativeArray Performance

**Unity.Collections.NativeArray:**
- Zero GC allocations (lives in native memory)
- Performance gains: 10x reported for certain collection operations
- Thread-safe for Job System
- Memory cost: Manual allocation/disposal required

---

## Critical Gotchas

### 1. Raycasting Against Wrong Collider Type

**Problem:** GPU raycasting (compute shaders) works against MeshRenderers, not physics colliders
- Physics.Raycast = colliders
- GPU raycast = mesh geometry
- Results don't match between CPU and GPU approaches

**Impact:** Can't mix-and-match GPU and CPU raycasting freely. Choose one architecture.

### 2. Frame Latency in Async Systems

**Problem:** RaycastCommand.ScheduleBatch introduces 1-frame delay
- Command submitted frame N
- Results available frame N+1
- Agent acts on data from previous frame

**Mitigation:** Acceptable for most sensor systems (human reaction time >> 16ms). Problematic for twitchy gameplay.

### 3. Cone Detection with SphereCast

**Problem:** SphereCast/RayCast cannot detect conical areas directly
- SphereCast finds sphere intersection
- Must manually filter results by angle
- Common source of bugs (forgetting angle check)

**Solution:** Always combine SphereCast with angle validation:
```csharp
if (Physics.SphereCast(...)) {
    float angle = Vector3.Angle(direction, hitPoint - origin);
    if (angle <= coneAngle) { /* valid */ }
}
```

### 4. Allocation in Update/FixedUpdate

**Problem:** Standard Physics.Raycast allocates memory every call
- Heap allocations trigger garbage collection
- GC pauses cause frame hitches
- Unacceptable for sensors querying every frame

**Solution:**
- Use `Physics.RaycastNonAlloc` with pre-allocated arrays
- Use `Unity.Collections.NativeArray` for Job System
- Set `NativeArrayOptions.UninitializedMemory` when safe

### 5. Layer Mask Mistakes

**Problem:** Forgetting layer masks makes raycasts test ALL colliders
- Massive performance cost
- One of top 3 raycast optimization mistakes

**Solution:** Always use layer masks:
```csharp
LayerMask sensorTargets = LayerMask.GetMask("Ships", "Obstacles");
Physics.Raycast(origin, direction, maxDistance, sensorTargets);
```

### 6. Underwater Acoustic Complexity

**Problem:** Realistic underwater acoustics involve:
- Sound speed profile changes (temperature/salinity/pressure)
- Refraction in water column
- Multi-path propagation (surface/bottom bounces)
- Shallow water waveguide effects

**Reality Check:** Professional tools (Bellhop3D, KRAKEN) are research-grade software requiring environmental meshes, sound speed profiles, bottom composition data.

**For Games:** Use simplified ray-based occlusion with material absorption. Perfect accuracy is neither achievable nor necessary for gameplay.

---

## Trade-off Analysis

### CPU vs GPU Raycasting

| Aspect | CPU (Physics.Raycast) | GPU (Compute Shader) |
|--------|----------------------|----------------------|
| **Targets** | Physics colliders | Mesh geometry |
| **Parallelism** | Job System (CPU cores) | GPU threads |
| **Latency** | Same frame (sync) or 1-frame (async) | 1+ frames (GPU→CPU readback) |
| **Performance** | 100s of rays/frame | 1000s+ of rays/frame |
| **Use Case** | Gameplay sensors, AI | Visual effects, sonar visualization |
| **Example** | Detection raycast | Sonar ping rendering |

**Recommendation:** CPU for gameplay-critical sensors, GPU for visual effects only.

### Spatial Partitioning Structures

| Structure | Dynamic Scenes | Static Scenes | Memory | Rebuild Cost |
|-----------|---------------|---------------|--------|--------------|
| **BVH** | Excellent | Good | Medium | Low (refit nodes) |
| **Octree** | Poor | Excellent | High | High (rebuild tree) |
| **Spatial Hash** | Good | Excellent | Low | Medium (rehash) |
| **Unity Built-in** | Excellent | Excellent | Hidden | Hidden |

**Recommendation:** Use Unity's built-in physics spatial partitioning. Only implement custom structures for non-physics queries (e.g., thermal field sampling).

### Sonar Resolution vs Performance

| Resolution | Rays per Ping | Frame Budget | Use Case |
|-----------|---------------|--------------|----------|
| Low (16x16) | 256 | <1ms | Minimap indicator |
| Medium (32x32) | 1,024 | ~5ms | Basic gameplay |
| High (64x64) | 4,096 | ~20ms | Visual sonar display |
| Research (128x128) | 16,384 | GPU only | Scientific accuracy |

**Critical Pattern:** Don't render what you don't need. Gameplay logic may use low-res sonar (cheap queries), while visual representation uses high-res GPU rendering (decoupled).

### Acoustic Simulation Approaches

| Approach | Accuracy | Performance | Complexity | Best For |
|----------|----------|-------------|------------|----------|
| **Ray-based occlusion** | Low | Excellent | Low | Real-time gameplay |
| **Image-source method** | Medium | Good | Medium | Indoor/small scenes |
| **Digital Huygens Model** | High | Medium (GPU) | High | Cinematic audio |
| **Wave equation solver** | Highest | Poor | Highest | Offline rendering |
| **Hybrid (rays + corrections)** | Medium-High | Good | Medium | AAA games |

**Recommendation for Underwater Game:** Ray-based with material absorption coefficients. Add simple multi-path (1-2 bounces max) if needed for gameplay depth.

---

## Red Flags

### 1. "Simple Raycasting Will Work"

**Warning:** Anyone suggesting "just use Physics.Raycast in Update()" for sonar has not implemented it.
- Real-world reports: Thousands of rays needed for acceptable resolution
- Performance craters without batching/Job System
- This is the #1 naive approach that fails in production

### 2. "Use Compute Shaders for Everything"

**Warning:** Compute shader raycasting has fundamental limitations
- Raycasts mesh geometry, not colliders (collision detection won't match)
- GPU→CPU readback is expensive (defeats the purpose)
- Only viable for pure rendering effects, not gameplay logic

### 3. "Octrees Are Always Faster"

**Warning:** Octrees are terrible for dynamic scenes
- Submarine game = everything moves (ship, fish, projectiles)
- Rebuilding octree every frame is slower than linear search for <1000 objects
- BVH or spatial hash are better for dynamic content

### 4. "We'll Use Real Acoustic Physics"

**Warning:** Research-grade acoustic simulation is not real-time
- KRAKEN, Bellhop3D require minutes/hours per scenario
- These tools need environmental data games don't have (full ocean sound speed profiles)
- Unrealistic expectations lead to scope creep and failed milestones

### 5. Missing Layer Masks

**Warning:** If example code doesn't show layer masks, it's incomplete
- "Works on my machine" with 10 objects in scene
- Fails in production with 1000+ colliders
- This is a litmus test for whether the author has shipped a game

### 6. Allocating Collections in Tight Loops

**Warning:** Code showing `new List<>()` or standard `Physics.Raycast` in Update
- Guaranteed GC pressure and frame hitches
- Shows lack of production experience
- Look for NativeArray and NonAlloc variants in real examples

---

## Libraries and Code Examples

### Production-Ready Packages

**1. SensorToolkit (Asset Store - Commercial)**
- URL: https://www.micosmo.com/sensortoolkit/
- Features: Vision sensors, range sensors, trigger sensors
- Optimized for performance with LOD support
- Used in shipped games

**2. Unity ML-Agents Grid Sensor**
- Repository: mbaske/grid-sensor
- Features: 3D grid-based sensors with frustum culling
- Shape scanning utilities for FOV detection
- Performance-focused design for ML training

**3. Graphics-Raycast (Open Source)**
- Repository: staggartcreations/Graphics-Raycast
- GPU-based raycasting against MeshRenderers
- Use case: Skinned mesh raycasting without baking
- Caveat: GPU→CPU readback latency

### Example Implementations

**4. Unity Detection Sensor**
- Repository: muveso/Unity-Detection-Sensor
- Object detection and optimized raycast system
- Example code for detection rays

**5. 2D Field of View**
- Repository: arthurgonze/2D-Field-of-View
- FOV and detection mechanic for top-down stealth
- Three test scenes demonstrating vision interactions
- Unity 2020.3 compatible

**6. Unity Sensory System**
- Repository: ntk4/UnitySensorySystem
- Modular system with Vision and Hearing senses
- Multiple view cones per NPC
- Configurable FoV in degrees

### Research Projects (Open Source)

**7. URSim - Underwater Vehicle Simulator**
- Repository: srmauvsoftware/URSim
- ROS + Unity3D integration
- Side-scan sonar and multibeam sonar
- Research paper: "Open Source Simulator for Unmanned Underwater Vehicles"

**8. Sonar Simulation**
- Repository: RomeldaB/Sonar-Simulation
- Ray tracing-based sonar imaging
- Unity 2019.4.25
- Demonstrates basic sonar principles

**9. Unity Batch Raycasting**
- Repository: adammyhre/Unity-Batch-Raycasting
- Demonstrates Job System + Burst Compiler
- RaycastCommand examples
- Performance-focused tutorial

### Academic/Research Tools

**10. GSound**
- URL: http://gamma.cs.unc.edu/GSOUND/
- Interactive sound propagation for games
- Reflection and diffraction via ray-based queries
- First practical real-time system

**11. Project Acoustics (Microsoft Research)**
- URL: https://www.microsoft.com/en-us/research/project/project-triton/
- Wave-based acoustic simulation
- Automatic occlusion and reverberation
- Production integration available

**12. Fuse - ROS Sensor Fusion**
- Repository: locusrobotics/fuse
- General sensor fusion architecture
- Nonlinear least squares optimization
- Production robotics use

**13. ROS Sensor Fusion Tutorial**
- Repository: methylDragon/ros-sensor-fusion-tutorial
- Step-by-step implementation guide
- Extended Kalman filter examples
- Beginner-friendly

---

## Applicability to Underwater Sensor Simulation

### Sonar Implementation Roadmap

**Phase 1: Gameplay Sensors (CPU-based)**
```
Goal: Low-res detection for AI and player feedback
- Physics.OverlapSphere for broad-phase (range detection)
- Angle filtering for cone FOV
- Batched raycasts (RaycastCommand) for occlusion
- Resolution: 16x16 to 32x32 beams
- Performance: <5ms per sonar ping
```

**Phase 2: Visual Sonar Display (GPU-based, decoupled)**
```
Goal: High-res sonar screen rendering
- Compute shader with 64x64+ rays
- Render to texture for UI display
- Update at lower frequency (e.g., 10 Hz instead of 60 Hz)
- Asynchronous update (1-2 frame latency acceptable)
```

**Phase 3: Acoustic Feedback (Audio system)**
```
Goal: Sonar pings and ambient sound
- Ray-based occlusion for direct path
- Material-based absorption (simple coefficients)
- Optional: 1-bounce reflections for "echo" effect
- Integration with Unity Audio or Wwise
```

### Thermal Sensor Implementation

**Approach: 3D Texture Field Sampling**
```csharp
// Setup: 3D texture representing thermal field
RenderTexture thermalField = new RenderTexture(64, 64, 64, RenderTextureFormat.ARGBFloat);
thermalField.dimension = TextureDimension.Tex3D;
thermalField.volumeDepth = 64;
thermalField.enableRandomWrite = true;

// Update: Compute shader for heat diffusion
ComputeShader heatDiffusion;
heatDiffusion.SetTexture(kernelHandle, "ThermalField", thermalField);
heatDiffusion.Dispatch(kernelHandle, 64/8, 64/8, 64/8);

// Query: Sample at world position
Vector3 uvw = (worldPos - fieldOrigin) / fieldSize;
// Use shader or CPU sampling to get temperature
```

**Performance:** 64³ voxels = 262,144 voxels, updateable at 60 FPS on GPU

**Trade-off:** Resolution vs spatial coverage. 64³ over 1000m³ ocean = 15.6m per voxel (coarse). May need adaptive resolution or multiple grids at different LODs.

### Multi-Sensor Fusion Architecture

**Pattern: ECS-based Sensor Systems (from research)**
```
SensorComponent (data only):
- SensorType (sonar/thermal/acoustic)
- Range, FOV, resolution
- Last update time
- Detection results (NativeArray)

SensorSystem (logic):
- Queries all SensorComponents
- Schedules batch jobs (raycasts, field samples)
- Writes results to components
- Fires events for detections

FusionSystem (higher-level):
- Reads multiple sensor results
- Combines detections (sensor fusion)
- Updates AI knowledge base
```

**Benefit:** Decoupling allows different update frequencies per sensor type
- Sonar: 30 Hz (gameplay critical)
- Thermal: 10 Hz (slow environmental change)
- Acoustic: Event-driven (only on ping)

### Performance Budget Recommendations

Based on research findings, for 60 FPS (16.67ms frame budget):

| System | Budget | Approach |
|--------|--------|----------|
| Sonar (per active sub) | 3-5ms | RaycastCommand batch (1024 rays) |
| Thermal sampling | 1ms | 3D texture lookup (GPU) |
| Acoustic propagation | 2ms | Ray occlusion + 1 bounce |
| Sensor fusion | 1ms | ECS system, data aggregation |
| **Total** | **7-9ms** | Leaves 7-9ms for rendering/gameplay |

**Scaling:** For multiple submarines, stagger sensor updates across frames (not all subs ping same frame).

---

## Recommended Architecture

### Hybrid Approach (Best Practices Synthesis)

**1. Broad-Phase Spatial Query**
- Unity's built-in physics partitioning (free)
- Physics.OverlapSphere for range detection
- Layer masks to filter irrelevant objects

**2. Mid-Phase Geometric Filtering**
- Angle check for cone/FOV sensors
- Distance-based LOD (far objects = lower ray counts)
- Frustum culling for directional sensors

**3. Narrow-Phase Raycasting**
- RaycastCommand + Job System for batching
- NativeArray for zero-allocation storage
- Non-blocking execution (continue with other work)

**4. Sensor Noise and Uncertainty**
- Perlin noise (3D) for natural variation
- Gaussian noise for sensor error
- Combined in post-process step after raycast results

**5. Visual Decoupling**
- Gameplay sensors: Low-res, high-frequency, CPU
- Visual sensors: High-res, low-frequency, GPU
- Never block gameplay on visual updates

**6. ECS Integration (Future-Proof)**
- SensorComponent: Data (range, FOV, results)
- SensorSystem: Logic (batching, scheduling)
- Easy to add new sensor types without code coupling

### Technology Stack Recommendations

**For Unity-Based Underwater Sensor Sim:**

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Core Raycasting** | RaycastCommand + Job System | 10x perf, proven in production |
| **Collections** | Unity.Collections.NativeArray | Zero GC, thread-safe |
| **Spatial Queries** | Physics.OverlapSphere | Built-in, optimized, reliable |
| **Thermal Fields** | 3D RenderTexture + Compute Shader | GPU acceleration, scales well |
| **Acoustic** | Ray-based + Unity Audio | Good-enough accuracy, real-time |
| **Architecture** | ECS (Unity DOTS) or modular systems | Decoupling, scalability |
| **Noise** | Perlin3D + Gaussian | Industry standard for sensor sim |

**Alternative for Non-Unity:**
- Godot: GDScript/C# with custom raycast batching
- Unreal: Blueprint + C++ with async traces
- Custom Engine: Embree (Intel ray tracing library) for CPU, OptiX for GPU

---

## Open Questions and Future Research

### Questions This Research Answers
1. How to efficiently raycast for sonar? **RaycastCommand + Job System**
2. GPU vs CPU for sensors? **CPU for gameplay, GPU for visuals**
3. Best spatial structure? **BVH (built-in) for dynamic, octree for static**
4. How to avoid GC? **NativeArray + NonAlloc variants**
5. Realistic acoustics? **Too expensive; use ray-based approximation**

### Questions Requiring Project-Specific Testing
1. **Optimal sonar resolution for gameplay feel?** (16x16 vs 32x32 vs 64x64)
   - Needs playtesting with actual submarine controls
   - Trade-off: Precision vs performance vs screen-space pixels

2. **Update frequency per sensor type?** (30Hz? 60Hz? Event-driven?)
   - Depends on gameplay pacing
   - Real sonar: ~1 ping/second; games may need faster for responsiveness

3. **Thermal field coverage vs resolution?** (Large area low-res vs small area high-res?)
   - Depends on gameplay: Is thermal used for detection or navigation?
   - May need multiple cascaded grids (near=high-res, far=low-res)

4. **Sensor fusion algorithm?** (Simple OR logic vs probabilistic fusion?)
   - How do multiple sensors combine? (Sonar + thermal + acoustic)
   - Bayesian fusion vs simple threshold-based?

5. **Noise parameters?** (How much Perlin vs Gaussian? What frequencies?)
   - Subjective: Does it "feel" like sonar to players?
   - May need audio designer input for acoustic noise character

### Gaps in External Research
1. **No complete underwater sensor game on GitHub** - examples are fragments
2. **Limited performance data for combined systems** - research tests sensors in isolation
3. **No standard sensor fusion architecture for games** - robotics solutions (ROS) are overkill
4. **Underwater-specific challenges under-documented** - refraction, multi-path, thermoclines

**Recommendation:** Prototype core sensor loop first, measure performance, iterate based on data.

---

## References and Further Reading

### Academic Papers
- "Physics-Based Modelling and Simulation of Multibeam Echosounder Perception for Autonomous Underwater Manipulation" (Frontiers in Robotics, 2021)
- "A rasterized ray-tracer pipeline for real-time, multi-device sonar simulation" (2020)
- "GSound: Interactive Sound Propagation for Games" (UNC Chapel Hill)
- "Real-time sound synthesis and propagation for games" (ResearchGate)

### Technical Articles
- "Unity Raycast Commands: A New Era" (TheGamedev.Guru)
- "Cull that cone! Improved cone/spotlight visibility tests" (Bart Wronski, 2017)
- "Optimizing spotlight intersection in tiled/clustered light culling" (Simon Coenen)
- "2D Visibility" (Red Blob Games)
- "Spatial Partition" (Game Programming Patterns)

### Documentation
- Unity Manual: Optimize raycasts and other physics queries
- Unity Scripting API: RaycastCommand, NativeArray
- Unity SensorSDK Architecture (v2.0.5)

### Open Source Projects
- srmauvsoftware/URSim (Underwater vehicle sim)
- mbaske/grid-sensor (ML-Agents sensors)
- adammyhre/Unity-Batch-Raycasting (Job System examples)
- locusrobotics/fuse (ROS sensor fusion)
- methylDragon/ros-sensor-fusion-tutorial (Kalman filter tutorial)

### Industry Tools
- Microsoft Project Acoustics
- Roblox Acoustic Simulation
- Wwise + Unreal Engine audio
- Energy2D (Heat simulation - educational)

---

## Conclusion

Sensor simulation for underwater games is a **solved problem at the pattern level**, but requires **careful engineering** to execute well. The core insights:

1. **Don't reinvent spatial partitioning** - use Unity's built-in physics
2. **Batch everything** - RaycastCommand + Job System is mandatory for scale
3. **Decouple visuals from logic** - low-res gameplay sensors, high-res rendering
4. **Embrace approximations** - perfect physics is the enemy of real-time
5. **Profile early** - performance characteristics change dramatically with object count

The path forward is clear: Start with proven patterns (OverlapSphere → angle filter → batched raycasts), measure performance with realistic object counts, and iterate based on data. Avoid the siren song of "realistic acoustic simulation" - games need responsive, predictable sensors, not research-grade accuracy.

---

**Document Status:** Complete
**Next Steps:** Synthesize with internal codebase analysis (from codebase-researcher) to create implementation plan (technical-planner)
**Related Documents:**
- (Pending) `shipmind_codebase_internal_research_2025-09-30.md` (from codebase-researcher agent)
- (Pending) `sensor_simulation_implementation_plan_2025-09-30.md` (from technical-planner agent)
