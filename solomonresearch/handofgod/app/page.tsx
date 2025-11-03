import Link from 'next/link'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            Hand of God
          </h1>
          <p className="text-2xl text-gray-600 dark:text-gray-300 mb-8">
            High-Performance Network Visualization & Simulation Platform
          </p>
          <p className="text-lg text-gray-500 dark:text-gray-400 mb-12 max-w-2xl mx-auto">
            Analyze and visualize large-scale networks with 5,000-10,000+ nodes.
            Advanced clustering, simulation, and intervention capabilities for network research.
          </p>

          <div className="flex gap-4 justify-center mb-16">
            <Link
              href="/auth/signup"
              className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
            >
              Get Started
            </Link>
            <Link
              href="/auth/login"
              className="px-8 py-3 bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors font-semibold"
            >
              Login
            </Link>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-semibold mb-2">High Performance</h3>
              <p className="text-gray-600 dark:text-gray-400">
                WebGL-powered rendering for smooth visualization of 5,000-10,000 node networks
              </p>
            </div>

            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
              <div className="text-4xl mb-4">🔬</div>
              <h3 className="text-xl font-semibold mb-2">Advanced Analysis</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Clustering, community detection, and comprehensive network metrics
              </p>
            </div>

            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
              <div className="text-4xl mb-4">🎮</div>
              <h3 className="text-xl font-semibold mb-2">Simulations</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Run intervention and diffusion simulations with variable repetitions
              </p>
            </div>
          </div>

          <div className="mt-16 p-8 bg-blue-50 dark:bg-gray-800 rounded-lg">
            <h2 className="text-2xl font-bold mb-4">Features Similar to Gephi, Enhanced for Research</h2>
            <ul className="text-left max-w-2xl mx-auto space-y-2 text-gray-700 dark:text-gray-300">
              <li>✓ Multiple layout algorithms (Force-directed, Hierarchical, Circular)</li>
              <li>✓ Node and edge styling by attributes</li>
              <li>✓ Community detection and clustering</li>
              <li>✓ Comprehensive centrality metrics</li>
              <li>✓ Network intervention simulations</li>
              <li>✓ Export to multiple formats</li>
              <li>✓ Cloud-based collaboration</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  )
}
