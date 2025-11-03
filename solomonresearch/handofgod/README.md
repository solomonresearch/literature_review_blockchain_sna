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
├── lib/                         # Utilities and core logic
│   ├── supabase/               # Supabase integration
│   └── network/                # Network algorithms and rendering
├── public/                      # Static assets
└── styles/                      # Global styles
```

## AI Development Guide

**Important**: If you're an AI assistant (like Claude, GPT, etc.) working on this project, please read the comprehensive development guide first:

**[.ai-docs/AI_DEVELOPMENT_GUIDE.md](.ai-docs/AI_DEVELOPMENT_GUIDE.md)**

This guide contains:
- Complete architecture and design decisions
- Technology stack details
- Database schema and Supabase configuration
- Development workflow and git practices
- Implementation phases and best practices
- Performance guidelines and optimization strategies
- Security considerations

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

### Phase 1: Foundation ✅
- [x] Project structure and documentation
- [ ] Supabase authentication
- [ ] Dashboard mockup

### Phase 2: Core Visualization
- [ ] WebGL renderer implementation
- [ ] Basic network loading
- [ ] Interactive controls
- [ ] Layout algorithms

### Phase 3: Advanced Features
- [ ] Clustering and community detection
- [ ] Network metrics
- [ ] Multi-attribute visualization

### Phase 4: Simulation
- [ ] Intervention simulations
- [ ] Diffusion models
- [ ] Results visualization

### Phase 5: Performance & Scale
- [ ] Optimize for 10k+ nodes
- [ ] Advanced rendering techniques

## License

[Specify License]

## Contact

- **Repository**: https://github.com/solomonresearch/handofgod
- **Issues**: https://github.com/solomonresearch/handofgod/issues

---

**Built for large-scale network research and analysis**
