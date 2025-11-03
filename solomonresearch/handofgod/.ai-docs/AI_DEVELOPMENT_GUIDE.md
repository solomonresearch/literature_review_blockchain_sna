# Hand of God - AI Development Guide

## Project Overview

**Hand of God** is a high-performance, scalable network visualization and simulation platform designed to handle large-scale networks (5,000-10,000+ nodes) with advanced analytical capabilities similar to Gephi, but with enhanced simulation and intervention features.

### Core Objectives
- Visualize and analyze large-scale networks (5,000-10,000 nodes)
- Provide Gephi-like functionality (clustering, layouts, attribute mapping)
- Enable network intervention and simulation with variable repetitions/rounds
- Deliver high-performance, browser-based visualization
- Support collaborative research with Supabase backend

---

## Technology Stack

### Frontend
- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Visualization**:
  - Primary: WebGL-based (Three.js, Sigma.js, or custom WebGL)
  - Alternative: D3.js for smaller networks
  - Canvas API for performance-critical rendering
- **State Management**: Zustand or React Context
- **UI Components**: shadcn/ui or Radix UI

### Backend & Database
- **Authentication**: Supabase Auth
- **Database**: Supabase (PostgreSQL)
- **Real-time**: Supabase Realtime
- **Storage**: Supabase Storage (for network files, exports)

### Performance & Optimization
- **Rendering**: WebGL for hardware acceleration
- **Spatial Indexing**: R-tree or Quadtree for node lookup
- **LOD**: Level of Detail rendering for large networks
- **Web Workers**: Offload layout calculations
- **Virtual Rendering**: Only render visible nodes/edges

---

## Supabase Configuration

### Connection Details
- **Database Password**: `e3LL39m9gKt!JvS`
- **Repository**: https://github.com/solomonresearch/handofgod

### Database Schema (Initial)

```sql
-- Users table (managed by Supabase Auth)

-- Networks table
CREATE TABLE networks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  node_count INTEGER,
  edge_count INTEGER,
  data JSONB, -- Stores network structure
  metadata JSONB, -- Additional properties
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Simulations table
CREATE TABLE simulations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  network_id UUID REFERENCES networks(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  simulation_type TEXT, -- intervention, diffusion, etc.
  parameters JSONB,
  rounds INTEGER DEFAULT 100,
  repetitions INTEGER DEFAULT 10,
  results JSONB,
  status TEXT DEFAULT 'pending', -- pending, running, completed, failed
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE
);

-- Layouts table (save custom layouts)
CREATE TABLE layouts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  network_id UUID REFERENCES networks(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  layout_type TEXT, -- force-directed, hierarchical, circular, etc.
  positions JSONB, -- Node positions
  parameters JSONB, -- Layout parameters
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE networks ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE layouts ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view their own networks"
  ON networks FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own networks"
  ON networks FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own networks"
  ON networks FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own networks"
  ON networks FOR DELETE
  USING (auth.uid() = user_id);

-- Similar policies for simulations and layouts
```

---

## Core Features & Implementation

### 1. Network Visualization

#### Requirements
- Display 5,000-10,000 nodes smoothly (60 FPS target)
- Interactive pan, zoom, select
- Multiple layout algorithms:
  - Force-directed (ForceAtlas2, Fruchterman-Reingold)
  - Hierarchical
  - Circular
  - Grid
  - Geographic (if lat/lon data available)
- Node styling by attributes (size, color, shape)
- Edge styling (color, width, directed/undirected)
- Clustering and community detection visualization

#### Implementation Approach
```typescript
// Use WebGL for rendering
// Implement spatial indexing for selection
// Use Web Workers for layout calculations
// Implement LOD (Level of Detail) for large networks
```

### 2. Clustering & Analysis

#### Features
- Community detection algorithms:
  - Louvain
  - Leiden
  - Label Propagation
  - Modularity-based
- Hierarchical clustering
- Visual cluster representation
- Cluster statistics and metrics

### 3. Network Metrics & Attributes

#### Node Metrics
- Degree (in/out/total)
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality
- PageRank
- Clustering coefficient

#### Network Metrics
- Density
- Average path length
- Diameter
- Modularity
- Connected components

### 4. Simulation & Intervention

#### Simulation Types
- **Intervention Simulation**: Remove/add nodes or edges and observe effects
- **Diffusion Simulation**: Information/disease spread
- **Cascading Failures**: Network resilience testing
- **Random Walk**: Explore network traversal

#### Configuration
- Number of rounds: 1-10,000
- Number of repetitions: 1-1,000
- Intervention strategies:
  - Targeted (high-centrality nodes)
  - Random
  - Custom selection
- Visualization of results over time
- Statistical summaries

---

## Project Structure

```
handofgod/
├── .ai-docs/
│   └── AI_DEVELOPMENT_GUIDE.md          # This file
├── .next/                                 # Next.js build output
├── app/
│   ├── (auth)/
│   │   ├── login/
│   │   │   └── page.tsx                  # Login page
│   │   └── signup/
│   │       └── page.tsx                  # Signup page
│   ├── (dashboard)/
│   │   ├── layout.tsx                    # Dashboard layout
│   │   ├── page.tsx                      # Main dashboard
│   │   ├── networks/
│   │   │   ├── page.tsx                  # Networks list
│   │   │   ├── [id]/
│   │   │   │   └── page.tsx              # Network viewer
│   │   │   └── new/
│   │   │       └── page.tsx              # Create network
│   │   └── simulations/
│   │       ├── page.tsx                  # Simulations list
│   │       └── [id]/
│   │           └── page.tsx              # Simulation viewer
│   ├── api/
│   │   ├── auth/
│   │   │   └── [...nextauth]/
│   │   │       └── route.ts              # Auth API routes
│   │   ├── networks/
│   │   │   └── route.ts                  # Network CRUD
│   │   └── simulations/
│   │       └── route.ts                  # Simulation CRUD
│   ├── layout.tsx                        # Root layout
│   └── page.tsx                          # Landing page
├── components/
│   ├── auth/
│   │   ├── LoginForm.tsx
│   │   └── SignupForm.tsx
│   ├── dashboard/
│   │   ├── DashboardHeader.tsx
│   │   ├── DashboardSidebar.tsx
│   │   └── DashboardStats.tsx
│   ├── network/
│   │   ├── NetworkCanvas.tsx             # Main visualization component
│   │   ├── NetworkControls.tsx           # Zoom, pan, layout controls
│   │   ├── NetworkStats.tsx              # Network statistics panel
│   │   ├── NodeInspector.tsx             # Node details panel
│   │   └── EdgeInspector.tsx             # Edge details panel
│   ├── simulation/
│   │   ├── SimulationConfig.tsx          # Simulation parameters
│   │   ├── SimulationPlayer.tsx          # Playback controls
│   │   └── SimulationResults.tsx         # Results visualization
│   ├── ui/                               # Reusable UI components
│   └── layouts/
│       └── LayoutSelector.tsx            # Layout algorithm selector
├── lib/
│   ├── supabase/
│   │   ├── client.ts                     # Supabase client
│   │   ├── auth.ts                       # Auth helpers
│   │   └── database.types.ts             # Generated types
│   ├── network/
│   │   ├── algorithms/
│   │   │   ├── layout.ts                 # Layout algorithms
│   │   │   ├── clustering.ts             # Clustering algorithms
│   │   │   ├── metrics.ts                # Network metrics
│   │   │   └── simulation.ts             # Simulation engines
│   │   ├── renderer/
│   │   │   ├── webgl-renderer.ts         # WebGL renderer
│   │   │   └── spatial-index.ts          # Spatial indexing
│   │   ├── parsers/
│   │   │   ├── gexf.ts                   # GEXF parser
│   │   │   ├── graphml.ts                # GraphML parser
│   │   │   └── csv.ts                    # CSV parser
│   │   └── types.ts                      # Network type definitions
│   └── utils/
│       ├── performance.ts                # Performance utilities
│       └── workers/
│           ├── layout.worker.ts          # Layout worker
│           └── metrics.worker.ts         # Metrics worker
├── public/
│   └── assets/
├── styles/
│   └── globals.css
├── .env.local                            # Environment variables
├── .gitignore
├── next.config.js
├── package.json
├── tsconfig.json
└── README.md
```

---

## Development Workflow

### CRITICAL: Git Workflow
**After EVERY interaction, new feature, or update, you MUST:**

1. **Add all changes**: `git add .`
2. **Commit with descriptive message**:
   ```bash
   git commit -m "feat: [concise description of changes]"
   # or
   git commit -m "fix: [bug fix description]"
   # or
   git commit -m "refactor: [refactoring description]"
   ```
3. **Push to origin**:
   ```bash
   git push origin master
   ```

### Commit Message Format
Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Example Workflow
```bash
# After implementing authentication
git add .
git commit -m "feat: implement Supabase authentication with login/signup pages"
git push origin master

# After adding network visualization
git add .
git commit -m "feat: add WebGL-based network visualization with 10k node support"
git push origin master

# After fixing a bug
git add .
git commit -m "fix: resolve node selection issue in clustered view"
git push origin master
```

---

## Implementation Phases

### Phase 1: Foundation (Current)
- [x] Create AI development guide
- [ ] Initialize Next.js project
- [ ] Set up Supabase authentication
- [ ] Create dashboard mockup
- [ ] Basic project structure

### Phase 2: Core Visualization
- [ ] Implement WebGL renderer
- [ ] Add basic network loading (CSV, JSON)
- [ ] Implement pan, zoom, selection
- [ ] Add basic layout algorithms (force-directed, circular)
- [ ] Node and edge styling

### Phase 3: Advanced Features
- [ ] Clustering algorithms
- [ ] Network metrics calculation
- [ ] Advanced layouts (hierarchical, geographic)
- [ ] Multi-attribute visualization
- [ ] Export functionality (PNG, SVG, PDF)

### Phase 4: Simulation System
- [ ] Simulation engine architecture
- [ ] Intervention simulations
- [ ] Diffusion models
- [ ] Results visualization
- [ ] Statistical analysis of results

### Phase 5: Performance & Scalability
- [ ] Optimize for 10k+ nodes
- [ ] Implement LOD system
- [ ] Add virtual rendering
- [ ] Performance profiling and optimization
- [ ] Memory management

### Phase 6: Collaboration & Export
- [ ] Shared networks (Supabase)
- [ ] Real-time collaboration
- [ ] Advanced export formats (GEXF, GraphML)
- [ ] Report generation

---

## Performance Guidelines

### Rendering Optimization
1. **Use WebGL** for all networks >1,000 nodes
2. **Spatial Indexing**: Implement R-tree or Quadtree
3. **Culling**: Don't render off-screen elements
4. **LOD**: Simplify distant nodes/edges
5. **Batching**: Batch WebGL draw calls
6. **Debouncing**: Debounce expensive operations (layout, metrics)

### Memory Management
1. **Lazy Loading**: Load network data on-demand
2. **Virtualization**: Only keep visible data in memory
3. **Web Workers**: Offload heavy computations
4. **Garbage Collection**: Be mindful of object creation

### Target Performance
- **60 FPS** for 5,000 nodes
- **30 FPS** for 10,000 nodes
- **<100ms** interaction response time
- **<2s** initial load time for 10k node network

---

## UI/UX Design Principles

### Dashboard Layout
```
┌─────────────────────────────────────────────────┐
│  Header (Logo, User, Search)                    │
├──────────┬──────────────────────────────────────┤
│          │                                       │
│ Sidebar  │  Main Canvas (Network Visualization) │
│          │                                       │
│ - Home   │                                       │
│ - Networks│                                      │
│ - Sims   │                                       │
│ - Stats  │                                       │
│          │                                       │
├──────────┴──────────────────────────────────────┤
│  Controls (Layout, Cluster, Style, Simulate)    │
└─────────────────────────────────────────────────┘
```

### Color Scheme
- Professional, academic aesthetic
- Dark mode support
- High contrast for accessibility
- Colorblind-friendly palettes

### Interaction Patterns
- **Click**: Select node/edge
- **Drag**: Pan canvas or move node
- **Scroll**: Zoom
- **Double-click**: Center on node
- **Right-click**: Context menu
- **Shift+Click**: Multi-select
- **Box select**: Lasso selection

---

## Security Considerations

1. **Authentication**: Always use Supabase Auth
2. **RLS**: Enable Row Level Security on all tables
3. **Validation**: Validate all user inputs
4. **Sanitization**: Sanitize network data uploads
5. **Rate Limiting**: Prevent abuse of simulation runs
6. **File Size Limits**: Limit network upload size
7. **CORS**: Configure properly for API access

---

## Testing Strategy

### Unit Tests
- Network algorithms
- Data parsers
- Metrics calculations

### Integration Tests
- Supabase integration
- Authentication flow
- Network CRUD operations

### Performance Tests
- 5k, 10k, 20k node rendering
- Layout algorithm benchmarks
- Simulation performance

### E2E Tests
- User registration and login
- Network creation and visualization
- Simulation execution

---

## Debugging & Monitoring

### Performance Monitoring
```typescript
// Use Chrome DevTools Performance tab
// Monitor:
// - Frame rate (should be 60 FPS)
// - Memory usage
// - JavaScript execution time
// - WebGL calls
```

### Error Tracking
- Use Sentry or similar for production
- Log errors to Supabase for analysis
- User feedback mechanism

---

## AI Assistant Guidelines

When working on this project, AI assistants should:

1. **Always create todo lists** for multi-step tasks
2. **Commit and push** after every feature/fix
3. **Reference this guide** for architectural decisions
4. **Maintain performance** standards (60 FPS target)
5. **Write TypeScript** with proper types
6. **Comment complex** algorithms
7. **Test at scale** (simulate with 5k-10k nodes)
8. **Follow the structure** defined in this guide
9. **Update this guide** when adding major features
10. **Ask for clarification** when requirements are ambiguous

---

## Resources & References

### Network Visualization Libraries
- **Sigma.js**: https://www.sigmajs.org/
- **Cytoscape.js**: https://js.cytoscape.org/
- **Three.js**: https://threejs.org/
- **D3.js**: https://d3js.org/

### Graph Algorithms
- **Graphology**: https://graphology.github.io/
- **NetworkX**: https://networkx.org/ (Python, for reference)

### Supabase
- **Docs**: https://supabase.com/docs
- **JavaScript Client**: https://supabase.com/docs/reference/javascript

### Performance
- **WebGL Best Practices**: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices
- **Web Workers**: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API

---

## Contact & Support

- **Repository**: https://github.com/solomonresearch/handofgod
- **Issues**: https://github.com/solomonresearch/handofgod/issues

---

## Version History

### v0.1.0 - Initial Setup (2025-11-03)
- Created AI Development Guide
- Defined project architecture
- Established development workflow
- Set up initial project structure

---

**Last Updated**: 2025-11-03
**Maintainer**: Solomon Research Team
