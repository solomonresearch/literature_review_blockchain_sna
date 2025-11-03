# Hand of God - Technical Architecture

> **Last Updated**: 2025-11-03
> **Version**: 0.1.0
> **Status**: 🚧 In Development

---

## 1. System Overview

Hand of God is a high-performance, web-based network visualization and simulation platform designed for analyzing network dynamics, diffusion processes, and behavioral propagation in large-scale networks (5,000-10,000+ nodes). Built with academic research, modularity, and extensibility in mind.

### Key Capabilities
- **Large-Scale Visualization**: WebGL-powered rendering for 5,000-10,000+ node networks
- **Network Simulation**: Threshold models, cascades, diffusion processes
- **Advanced Analytics**: Network metrics, centrality measures, community detection
- **Gephi-like Interface**: Familiar tools for researchers and analysts
- **Cloud-Powered**: Supabase backend for authentication, storage, and collaboration

---

## 2. Technology Stack

### Frontend
- **Framework**: Next.js 14+ with React 18 (TypeScript)
- **Visualization**:
  - D3.js v7 (custom layouts and data binding)
  - Three.js (WebGL rendering for large networks)
  - Canvas API (high-performance rendering)
  - Graphology (graph data structure and algorithms)
- **UI Components**: shadcn/ui (Radix UI primitives + Tailwind CSS)
- **Charts**: Recharts or Chart.js
- **State Management**: Zustand
- **Icons**: Lucide React
- **File Handling**: PapaParse (CSV), custom GEXF/GraphML parsers

### Backend & Database
- **Authentication**: Supabase Auth
- **Database**: Supabase (PostgreSQL)
- **Real-time**: Supabase Realtime
- **Storage**: Supabase Storage (network files, simulation results)

### Build Tools
- **Bundler**: Next.js (Webpack-based)
- **Package Manager**: npm
- **Linting**: ESLint + Prettier
- **Type Checking**: TypeScript 5.3+

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Import    │  │    Canvas   │  │  Analytics  │             │
│  │   Panel     │  │   Component │  │    Panel    │             │
│  │             │  │             │  │             │             │
│  │ • Upload UI │  │ • Network   │  │ • Charts    │             │
│  │ • Controls  │  │   Renderer  │  │ • Stats     │             │
│  │ • Settings  │  │ • Zoom/Pan  │  │ • Legend    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              State Management (Zustand)                   │  │
│  │  • networkState  • simulationState  • uiState             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Network   │  │  Simulation │  │  Analytics  │            │
│  │   Manager   │  │   Engine    │  │   Engine    │            │
│  │             │  │             │  │             │            │
│  │ • Graph ops │  │ • Rules     │  │ • Metrics   │            │
│  │ • Layouts   │  │ • Updates   │  │ • Tracking  │            │
│  │ • Attributes│  │ • Rounds    │  │ • Aggreg.   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                         DATA LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Parser    │  │    Graph    │  │   Export    │            │
│  │   Module    │  │    Model    │  │   Module    │            │
│  │             │  │             │  │             │            │
│  │ • CSV Parse │  │ • Nodes[]   │  │ • CSV Gen   │            │
│  │ • GEXF Parse│  │ • Edges[]   │  │ • GEXF Gen  │            │
│  │ • Validation│  │ • Attr Map  │  │ • JSON      │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└───────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    BACKEND LAYER (Supabase)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Auth        │  │ Database    │  │  Storage    │            │
│  │ Service     │  │ (PostgreSQL)│  │  Buckets    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Specifications

### 4.1 Parser Module
**Status**: ⏳ Not Started
**Location**: `lib/network/parsers/`

**Responsibilities:**
- Parse CSV, GEXF, GraphML, JSON input files
- Validate data format and structure
- Convert to internal Graphology graph representation
- Handle different network formats (edge list, adjacency matrix)

**Interface:**
```typescript
interface ParserModule {
  parseCSV(file: File): Promise<NetworkData>
  parseGEXF(file: File): Promise<NetworkData>
  parseGraphML(file: File): Promise<NetworkData>
  parseJSON(file: File): Promise<NetworkData>
  validateNetwork(data: NetworkData): ValidationResult
  detectFormat(file: File): NetworkFormat
}

interface NetworkData {
  nodes: Node[]
  edges: Edge[]
  metadata: NetworkMetadata
}

type NetworkFormat = 'csv' | 'gexf' | 'graphml' | 'json'
```

### 4.2 Network Manager Module
**Status**: ⏳ Not Started
**Location**: `lib/network/manager.ts`

**Responsibilities:**
- Store and manage Graphology graph instance
- Implement layout algorithms (via graphology-layout packages)
- Handle attribute updates
- Provide network queries

**Interface:**
```typescript
interface NetworkManager {
  loadNetwork(data: NetworkData): void
  getGraph(): Graph
  applyLayout(algorithm: LayoutAlgorithm): void
  setNodeAttribute(nodeId: string, attr: string, value: any): void
  calculateMetrics(): NetworkMetrics
}

type LayoutAlgorithm =
  | 'forceAtlas2'
  | 'circular'
  | 'noverlap'
  | 'random'
```

### 4.3 Simulation Engine Module
**Status**: ⏳ Not Started
**Location**: `lib/network/algorithms/simulation.ts`

**Responsibilities:**
- Execute simulation rules (threshold, cascade, diffusion)
- Manage simulation rounds and repetitions
- Track state changes across rounds
- Control simulation flow (play, pause, step, reset)

**Interface:**
```typescript
interface SimulationEngine {
  setRules(rules: SimulationRules): void
  start(): void
  pause(): void
  step(): void
  reset(): void
  executeRound(): SimulationResult
  getHistory(): RoundHistory[]
}

interface SimulationRules {
  model: 'threshold' | 'cascade' | 'diffusion' | 'custom'
  updateMode: 'synchronous' | 'asynchronous'
  targetAttribute: string
  parameters: Record<string, any>
  rounds: number
  repetitions: number
}
```

### 4.4 Visualization Renderer Module
**Status**: 🔄 In Progress (basic mockup exists)
**Location**: `components/network/NetworkCanvas.tsx`

**Responsibilities:**
- Render network using WebGL for performance
- Handle user interactions (zoom, pan, click, select)
- Update visual properties dynamically
- Provide visual feedback during simulation

**Interface:**
```typescript
interface VisualizationRenderer {
  render(graph: Graph, layout: LayoutPositions): void
  update(): void
  setColorMapping(attribute: string, colorScale: ColorScale): void
  setSizeMapping(attribute: string, sizeScale: SizeScale): void
  onNodeClick(callback: (node: Node) => void): void
  exportAsPNG(): Blob
}
```

### 4.5 Analytics Module
**Status**: ⏳ Not Started
**Location**: `lib/network/algorithms/metrics.ts`

**Responsibilities:**
- Calculate network metrics using graphology-metrics
- Track attribute distributions
- Generate charts and statistics
- Monitor simulation progress

**Interface:**
```typescript
interface AnalyticsModule {
  calculateMetrics(graph: Graph): NetworkMetrics
  getAttributeDistribution(attribute: string): Distribution
  trackAttributeOverTime(attribute: string): TimeSeries
  measureConvergence(): number
}

interface NetworkMetrics {
  nodeCount: number
  edgeCount: number
  avgDegree: number
  density: number
  diameter?: number
  clustering?: number
  modularity?: number
}
```

### 4.6 Export Module
**Status**: ⏳ Not Started
**Location**: `lib/network/parsers/export.ts`

**Responsibilities:**
- Export network to various formats (CSV, GEXF, JSON)
- Export statistics and charts
- Save simulation configuration
- Generate reports

**Interface:**
```typescript
interface ExportModule {
  exportNetworkCSV(graph: Graph): Blob
  exportNetworkGEXF(graph: Graph): Blob
  exportStatistics(metrics: NetworkMetrics[]): Blob
  saveConfiguration(config: SimulationConfig): Blob
  generateReport(options: ReportOptions): Blob
}
```

---

## 5. Data Flow

### Loading Network
```
User uploads file
  → ParserModule.parseCSV()
  → NetworkManager.loadNetwork()
  → VisualizationRenderer.render()
  → AnalyticsModule.calculateMetrics()
  → Save to Supabase (optional)
```

### Running Simulation
```
User configures simulation rules
  → SimulationEngine.setRules()
  → User clicks "Play"
  → SimulationEngine.start()
  → Web Worker executes rounds
    → Apply rules to each node
    → NetworkManager.setNodeAttribute()
    → VisualizationRenderer.update()
    → AnalyticsModule.trackAttributeOverTime()
  → Check convergence or max rounds
  → Save results to Supabase
```

### Exporting Results
```
User clicks "Export GEXF"
  → ExportModule.exportNetworkGEXF()
  → Browser downloads file
  → Upload to Supabase Storage (optional)
```

---

## 6. State Management (Zustand)

### Store Structure
```typescript
interface NetworkStore {
  graph: Graph | null
  layout: LayoutPositions
  selectedLayout: LayoutAlgorithm

  loadNetwork: (data: NetworkData) => void
  applyLayout: (algorithm: LayoutAlgorithm) => void
  updateNodeAttribute: (nodeId: string, attr: string, value: any) => void
}

interface SimulationStore {
  rules: SimulationRules | null
  currentRound: number
  maxRounds: number
  isRunning: boolean
  hasConverged: boolean
  history: RoundHistory[]

  setRules: (rules: SimulationRules) => void
  start: () => void
  pause: () => void
  step: () => void
  reset: () => void
}

interface UIStore {
  selectedNode: string | null
  hoveredNode: string | null
  colorAttribute: string | null
  sizeAttribute: string | null
  zoom: number
  pan: {x: number, y: number}

  setSelectedNode: (nodeId: string | null) => void
  setColorAttribute: (attr: string | null) => void
}
```

---

## 7. File Format Specifications

### Input Formats

**CSV - Edge List**
```csv
source,target,weight
node_1,node_2,1.0
node_2,node_3,0.8
```

**CSV - Node Attributes**
```csv
id,opinion,influence
node_1,positive,0.7
node_2,neutral,0.5
```

**GEXF (Gephi Exchange Format)**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
  <graph mode="static" defaultedgetype="undirected">
    <nodes>
      <node id="0" label="Node 1"/>
    </nodes>
    <edges>
      <edge id="0" source="0" target="1"/>
    </edges>
  </graph>
</gexf>
```

---

## 8. Performance Considerations

### Optimization Strategies

1. **Large Networks (>5000 nodes)**
   - ✅ Use WebGL rendering (Three.js)
   - Use spatial indexing (R-tree/Quadtree) for selection
   - Throttle render updates
   - Use Web Workers for layout calculations

2. **Layout Calculations**
   - Use graphology-layout-forceatlas2 with workers
   - Cache layout results
   - Allow early termination

3. **Simulation Performance**
   - Run simulation in Web Worker
   - Batch attribute updates
   - Only update changed nodes
   - Progressive rendering

4. **Memory Management**
   - Limit simulation history storage
   - Use virtual scrolling for large lists
   - Clear cached data periodically

**Performance Targets:**
- **60 FPS** for 5,000 nodes
- **30 FPS** for 10,000 nodes
- **<100ms** interaction response
- **<2s** initial load for 10k nodes

---

## 9. Database Schema (Supabase)

```sql
-- Networks table
CREATE TABLE networks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  node_count INTEGER,
  edge_count INTEGER,
  data JSONB,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Simulations table
CREATE TABLE simulations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  network_id UUID REFERENCES networks(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  simulation_type TEXT,
  parameters JSONB,
  rounds INTEGER DEFAULT 100,
  repetitions INTEGER DEFAULT 10,
  results JSONB,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE
);

-- Layouts table
CREATE TABLE layouts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  network_id UUID REFERENCES networks(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  layout_type TEXT,
  positions JSONB,
  parameters JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## 10. Implementation Status

| Module | Status | Priority | Notes |
|--------|--------|----------|-------|
| Parser Module | ⏳ Not Started | High | CSV, GEXF, GraphML parsers |
| Network Manager | ⏳ Not Started | High | Graphology integration |
| Visualization | 🔄 In Progress | High | Basic mockup exists |
| Simulation Engine | ⏳ Not Started | High | Core functionality |
| Analytics | ⏳ Not Started | Medium | Metrics calculation |
| Export Module | ⏳ Not Started | Medium | Multiple formats |
| Web Workers | ⏳ Not Started | High | Performance critical |
| Supabase Integration | ✅ Complete | High | Auth working |
| UI Components | 🔄 In Progress | High | shadcn/ui integrated |

---

## 11. Testing Strategy

### Unit Tests
- Parser module: various file formats
- Network manager: graph operations
- Simulation engine: rule execution
- Analytics: metric calculations

### Integration Tests
- File import → visualization
- Simulation run → export
- Layout change → re-render

### E2E Tests
- Complete workflow: import → simulate → export
- User interactions: zoom, pan, select
- Error handling: invalid files, large networks

---

## 12. Security Considerations

1. **File Upload**
   - Validate file size (<50MB)
   - Check file types (.csv, .gexf, .graphml, .json)
   - Client-side parsing only
   - Sanitize content before processing

2. **Custom Rules**
   - Sandbox JavaScript execution in Web Worker
   - Timeout long-running code
   - Validate syntax before execution

3. **Data Privacy**
   - Client-side processing where possible
   - Encrypted storage in Supabase
   - User data isolation (RLS policies)

---

## 13. Future Enhancements

### Phase 2
- Multi-attribute simulations
- Network generation (synthetic networks)
- Advanced community detection
- Real-time collaboration
- Python/R API integration

### Phase 3
- Machine learning integration
- Temporal network analysis
- Geographic network visualization
- Mobile app (React Native)

---

## References

- **Graphology**: https://graphology.github.io/
- **Gephi**: https://gephi.org/
- **Supabase**: https://supabase.com/docs
- **Next.js**: https://nextjs.org/docs
- **Three.js**: https://threejs.org/docs

---

**Maintained by**: Solomon Research Team
**Repository**: https://github.com/solomonresearch/handofgod
