import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

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
            <Button asChild size="lg">
              <Link href="/auth/signup">
                Get Started
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/auth/login">
                Login
              </Link>
            </Button>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <Card>
              <CardHeader>
                <div className="text-4xl mb-4">⚡</div>
                <CardTitle>High Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  WebGL-powered rendering for smooth visualization of 5,000-10,000 node networks
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="text-4xl mb-4">🔬</div>
                <CardTitle>Advanced Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Clustering, community detection, and comprehensive network metrics
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="text-4xl mb-4">🎮</div>
                <CardTitle>Simulations</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Run intervention and diffusion simulations with variable repetitions
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          <Card className="mt-16">
            <CardHeader>
              <CardTitle className="text-2xl">Features Similar to Gephi, Enhanced for Research</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-left max-w-2xl mx-auto space-y-2">
                <li>✓ Multiple layout algorithms (Force-directed, Hierarchical, Circular)</li>
                <li>✓ Node and edge styling by attributes</li>
                <li>✓ Community detection and clustering</li>
                <li>✓ Comprehensive centrality metrics</li>
                <li>✓ Network intervention simulations</li>
                <li>✓ Export to multiple formats</li>
                <li>✓ Cloud-based collaboration</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  )
}
