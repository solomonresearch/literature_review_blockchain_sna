import NetworkCanvas from '@/components/network/NetworkCanvas'
import NetworkControls from '@/components/network/NetworkControls'

export default function DashboardPage() {
  return (
    <div className="flex flex-col h-full">
      {/* Main Network Visualization Area */}
      <div className="flex-1 overflow-hidden">
        <NetworkCanvas />
      </div>

      {/* Controls */}
      <NetworkControls />

      {/* Side Panel for Node/Edge Inspector (optional) */}
      <div className="hidden xl:block absolute right-4 top-20 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 max-h-[calc(100vh-200px)] overflow-y-auto">
        <h3 className="font-semibold mb-4 text-lg">Inspector</h3>

        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Selection
            </h4>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Click on a node or edge to inspect its properties
            </p>
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Quick Actions
            </h4>
            <div className="space-y-2">
              <button className="w-full px-3 py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-md text-sm hover:bg-blue-100 dark:hover:bg-blue-900/30">
                Calculate Metrics
              </button>
              <button className="w-full px-3 py-2 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded-md text-sm hover:bg-purple-100 dark:hover:bg-purple-900/30">
                Detect Communities
              </button>
              <button className="w-full px-3 py-2 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded-md text-sm hover:bg-green-100 dark:hover:bg-green-900/30">
                Export Network
              </button>
            </div>
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Simulations
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Type:</span>
                <span className="font-medium">Intervention</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Rounds:</span>
                <input
                  type="number"
                  defaultValue="100"
                  className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-right dark:bg-gray-700"
                />
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Repetitions:</span>
                <input
                  type="number"
                  defaultValue="10"
                  className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-right dark:bg-gray-700"
                />
              </div>
              <button className="w-full mt-2 px-3 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-md font-medium">
                Configure & Run
              </button>
            </div>
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Recent Activity
            </h4>
            <div className="space-y-2 text-xs text-gray-600 dark:text-gray-400">
              <div className="flex items-start gap-2">
                <span>📊</span>
                <div>
                  <p>Calculated betweenness centrality</p>
                  <p className="text-gray-400 dark:text-gray-500">2 min ago</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span>🎮</span>
                <div>
                  <p>Ran diffusion simulation</p>
                  <p className="text-gray-400 dark:text-gray-500">5 min ago</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span>💾</span>
                <div>
                  <p>Saved network layout</p>
                  <p className="text-gray-400 dark:text-gray-500">10 min ago</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
