# Graph Report - .  (2026-07-15)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 98 nodes · 163 edges · 18 communities (9 shown, 9 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ce2a84df`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- MLPTrainer
- CellEngine
- MainActivity
- CellService
- .trainDistilled
- CellState
- Hexagram
- FloatArray
- .writeTensor
- CellEngine.kt
- DistilledMLPTrainer
- .tick
- .trainBatch
- gradlew

## God Nodes (most connected - your core abstractions)
1. `CellEngine` - 21 edges
2. `DistilledMLPTrainer` - 19 edges
3. `MLPTrainer` - 11 edges
4. `MainActivity` - 10 edges
5. `CellService` - 7 edges
6. `CellState` - 6 edges
7. `Hexagram` - 5 edges
8. `DistilledSample` - 4 edges
9. `getInstance()` - 3 edges
10. `DistillStatus` - 2 edges

## Surprising Connections (you probably didn't know these)
- `CellService` --references--> `CellEngine`  [EXTRACTED]
  app/src/main/java/com/operit/cellpet/CellService.kt → app/src/main/java/com/operit/cellpet/CellEngine.kt
- `MainActivity` --references--> `CellEngine`  [EXTRACTED]
  app/src/main/java/com/operit/cellpet/MainActivity.kt → app/src/main/java/com/operit/cellpet/CellEngine.kt

## Import Cycles
- None detected.

## Communities (18 total, 9 thin omitted)

### Community 0 - "MLPTrainer"
Cohesion: 0.31
Nodes (5): ByteOrder, FloatArray, IntArray, java, MLPTrainer

### Community 2 - "MainActivity"
Cohesion: 0.22
Nodes (6): MainActivity, AppCompatActivity, Bundle, Button, ProgressBar, TextView

### Community 3 - "CellService"
Cohesion: 0.25
Nodes (4): CellService, IBinder, Intent, Service

### Community 4 - ".trainDistilled"
Cohesion: 0.39
Nodes (4): DecisionRule, DistilledSample, DistillStatus, TrainResult

### Community 9 - "CellEngine.kt"
Cohesion: 0.50
Nodes (3): ExperienceRecord, getInstance(), Context

## Knowledge Gaps
- **1 isolated node(s):** `ExperienceRecord`
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CellEngine` connect `CellEngine` to `MainActivity`, `CellService`, `CellState`, `CellEngine.kt`, `.getDistilledRules`, `.tick`?**
  _High betweenness centrality (0.400) - this node is a cross-community bridge._
- **Why does `DistilledMLPTrainer` connect `DistilledMLPTrainer` to `CellEngine`, `.trainDistilled`, `FloatArray`, `.writeTensor`, `.getDistilledRules`, `.trainBatch`?**
  _High betweenness centrality (0.273) - this node is a cross-community bridge._
- **Why does `MainActivity` connect `MainActivity` to `CellEngine`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **What connects `ExperienceRecord` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._