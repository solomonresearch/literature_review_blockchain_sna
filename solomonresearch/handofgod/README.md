# Hand of God

A high-performance network visualization and simulation platform for large-scale network analysis.

## Overview

**Hand of God** is designed to handle large networks (5,000-10,000+ nodes) with advanced visualization, clustering, and simulation capabilities similar to Gephi, but with enhanced features for network intervention and simulation analysis.

### Key Features

- **High-Performance Visualization**: WebGL-based rendering for smooth visualization of networks with 5,000-10,000+ nodes
- **Advanced Layouts**: Multiple layout algorithms (force-directed, hierarchical, circular, grid)
- **Clustering & Analysis**: Community detection, modularity analysis, and visual clustering
- **Network Metrics**: Comprehensive centrality measures, clustering coefficients, and network statistics
- **Intervention & Simulation**: Run network simulations with variable repetitions and rounds to study network dynamics
- **Gephi-like Interface**: Familiar controls for display, styling, and attribute mapping
- **Cloud-Powered**: Supabase authentication and database for secure, collaborative research

## Technology Stack

- **Frontend**: Next.js 14+ with TypeScript
- **Visualization**: WebGL (Three.js/Sigma.js) for high-performance rendering
- **Backend**: Supabase (Authentication, PostgreSQL Database, Real-time)
- **Styling**: Tailwind CSS
- **State Management**: Zustand/React Context

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Supabase account
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/solomonresearch/handofgod.git
cd handofgod

# Install dependencies
npm install
# or
yarn install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your Supabase credentials

# Run development server
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
handofgod/
├── .ai-docs/                    # AI assistant development guide
│   └── AI_DEVELOPMENT_GUIDE.md  # Comprehensive project documentation
├── app/                         # Next.js app directory
├── components/                  # React components
│   ├── auth/                   # Authentication components
│   ├── dashboard/              # Dashboard layout
│   ├── network/                # Network visualization
│   ├── simulation/             # Simulation controls
│   └── ui/                     # shadcn/ui components
├── lib/                         # Utilities and core logic
│   ├── supabase/               # Supabase integration
│   ├── network/                # Network algorithms and rendering
│   │   ├── algorithms/         # Layout, metrics, simulation
│   │   ├── parsers/            # File format parsers
│   │   └── renderer/           # WebGL renderer
│   └── utils/                  # Helper functions
├── public/                      # Static assets
├── styles/                      # Global styles
├── ARCHITECTURE.md              # Technical architecture
└── TODO.md                      # Implementation checklist
```

## Documentation

### For AI Assistants & Developers

**Important**: Read these documents before contributing:

1. **[.ai-docs/AI_DEVELOPMENT_GUIDE.md](.ai-docs/AI_DEVELOPMENT_GUIDE.md)** - Complete development guide
   - Technology stack details
   - Database schema and Supabase configuration
   - Development workflow and git practices
   - Performance guidelines and optimization strategies
   - Security considerations

2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture
   - System overview and design
   - Module specifications and interfaces
   - Data flow and state management
   - File format specifications
   - Implementation status

3. **[TODO.md](TODO.md)** - Implementation checklist
   - Comprehensive task list (100+ tasks)
   - Organized by module and phase
   - Priority matrix and milestones
   - Progress tracking

### Architecture Overview

Hand of God follows a modular architecture with clear separation of concerns:

**Presentation Layer** → **Application Layer** → **Data Layer** → **Backend (Supabase)**

#### Core Modules

1. **Parser Module** (`lib/network/parsers/`)
   - Supports CSV, GEXF, GraphML, JSON formats
   - Validates and converts to Graphology graph format

2. **Network Manager** (`lib/network/manager.ts`)
   - Manages graph data structure using Graphology
   - Handles layout algorithms (ForceAtlas2, circular, etc.)
   - Node/edge attribute management

3. **Visualization Renderer** (`components/network/NetworkCanvas.tsx`)
   - WebGL-based rendering for performance
   - Spatial indexing for large networks
   - Interactive zoom, pan, selection

4. **Simulation Engine** (`lib/network/algorithms/simulation.ts`)
   - Threshold, cascade, diffusion models
   - Web Worker-based execution
   - Rounds and repetitions support

5. **Analytics Module** (`lib/network/algorithms/metrics.ts`)
   - Network metrics (centrality, clustering, etc.)
   - Community detection (Louvain algorithm)
   - Time series tracking

6. **Export Module** (`lib/network/parsers/export.ts`)
   - Export to CSV, GEXF, GraphML, JSON
   - Statistical reports and charts
   - Configuration save/load

## Features

### Network Visualization
- Load and display networks from multiple formats (GEXF, GraphML, CSV, JSON)
- Interactive pan, zoom, and selection
- Multiple layout algorithms
- Style nodes and edges by attributes (size, color, shape, width)
- Cluster visualization with community detection

### Simulation & Intervention
- Run intervention simulations on network structure
- Diffusion and cascade modeling
- Configure repetitions (1-1,000) and rounds (1-10,000)
- Visualize simulation results over time
- Statistical analysis and export

### Performance
- Optimized for 5,000-10,000 node networks
- WebGL hardware acceleration
- Spatial indexing for fast interaction
- Web Workers for background computation
- Level of Detail (LOD) rendering

## Development Workflow

### Git Workflow (CRITICAL)

After **every interaction, new feature, or update**, commit and push changes:

```bash
# Add all changes
git add .

# Commit with descriptive message (use conventional commits)
git commit -m "feat: description of changes"

# Push to origin
git push origin master
```

### Commit Message Format
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance

## Database Setup

The project uses Supabase for authentication and data storage. See [AI_DEVELOPMENT_GUIDE.md](.ai-docs/AI_DEVELOPMENT_GUIDE.md) for detailed database schema and setup instructions.

## Performance Targets

- **60 FPS** for 5,000 nodes
- **30 FPS** for 10,000 nodes
- **<100ms** interaction response time
- **<2s** initial load for 10k node network

## Contributing

1. Read the [AI Development Guide](.ai-docs/AI_DEVELOPMENT_GUIDE.md)
2. Create a feature branch
3. Make your changes
4. Commit using conventional commits format
5. Push and create a pull request

## Roadmap

> **See [TODO.md](TODO.md) for detailed implementation checklist**

### Phase 1: Foundation (Week 1-4) 🔄
- [x] Project structure and documentation
- [x] Supabase authentication
- [x] Dashboard mockup with shadcn/ui
- [ ] CSV parser implementation ⚠️
- [ ] Basic network visualization ⚠️
- [ ] Graphology integration ⚠️

### Phase 2: Core Visualization (Week 5-8)
- [ ] WebGL renderer implementation
- [ ] All file format parsers (CSV, GEXF, GraphML, JSON)
- [ ] Interactive controls (zoom, pan, select)
- [ ] Layout algorithms (ForceAtlas2, circular, etc.)
- [ ] Color and size mapping

### Phase 3: Simulation Engine (Week 9-12)
- [ ] Simulation engine core
- [ ] Threshold, cascade, diffusion models
- [ ] Web Workers for performance
- [ ] Simulation UI and controls
- [ ] Results tracking and export

### Phase 4: Analytics (Week 13-16)
- [ ] Network metrics (centrality, clustering, etc.)
- [ ] Community detection (Louvain)
- [ ] Charts and visualizations
- [ ] Time series analysis
- [ ] Statistical reports

### Phase 5: Advanced Features & Production (Week 17+)
- [ ] Performance optimization (10k+ nodes @ 60fps)
- [ ] Network generation tools
- [ ] Collaboration features
- [ ] API and integrations
- [ ] Comprehensive testing
- [ ] Production deployment

## License

[Specify License]

## Contact

- **Repository**: https://github.com/solomonresearch/handofgod
- **Issues**: https://github.com/solomonresearch/handofgod/issues

---

**Built for large-scale network research and analysis**
