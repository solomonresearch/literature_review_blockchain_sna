'use client'

import { useState } from 'react'

export default function NetworkControls() {
  const [layout, setLayout] = useState('force-directed')
  const [showClusters, setShowClusters] = useState(false)
  const [nodeSize, setNodeSize] = useState(5)

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center gap-8">
        {/* Layout Selection */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Layout:
          </label>
          <select
            value={layout}
            onChange={(e) => setLayout(e.target.value)}
            className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="force-directed">Force-Directed</option>
            <option value="hierarchical">Hierarchical</option>
            <option value="circular">Circular</option>
            <option value="grid">Grid</option>
            <option value="geographic">Geographic</option>
          </select>
        </div>

        {/* Clustering Toggle */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Clusters:
          </label>
          <button
            onClick={() => setShowClusters(!showClusters)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              showClusters
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            {showClusters ? 'On' : 'Off'}
          </button>
        </div>

        {/* Node Size Slider */}
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
            Node Size:
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={nodeSize}
            onChange={(e) => setNodeSize(parseInt(e.target.value))}
            className="flex-1"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400 w-8">
            {nodeSize}
          </span>
        </div>

        {/* Display Options */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Display:
          </label>
          <div className="flex gap-1">
            <button
              className="px-3 py-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-md text-sm"
              title="Show Labels"
            >
              🏷️
            </button>
            <button
              className="px-3 py-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-md text-sm"
              title="Color by Attribute"
            >
              🎨
            </button>
            <button
              className="px-3 py-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-md text-sm"
              title="Filter"
            >
              🔍
            </button>
          </div>
        </div>

        {/* Simulation Button */}
        <button className="ml-auto px-4 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded-md text-sm font-medium">
          ▶ Run Simulation
        </button>
      </div>
    </div>
  )
}
