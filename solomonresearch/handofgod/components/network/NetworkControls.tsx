'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'

export default function NetworkControls() {
  const [layout, setLayout] = useState('force-directed')
  const [showClusters, setShowClusters] = useState(false)
  const [nodeSize, setNodeSize] = useState([5])

  return (
    <div className="bg-card border-t border-border p-4">
      <div className="flex items-center gap-8 flex-wrap">
        {/* Layout Selection */}
        <div className="flex items-center gap-2">
          <Label className="text-sm font-medium whitespace-nowrap">
            Layout:
          </Label>
          <Select value={layout} onValueChange={setLayout}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select layout" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="force-directed">Force-Directed</SelectItem>
              <SelectItem value="hierarchical">Hierarchical</SelectItem>
              <SelectItem value="circular">Circular</SelectItem>
              <SelectItem value="grid">Grid</SelectItem>
              <SelectItem value="geographic">Geographic</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Clustering Toggle */}
        <div className="flex items-center gap-2">
          <Label className="text-sm font-medium whitespace-nowrap">
            Clusters:
          </Label>
          <Button
            onClick={() => setShowClusters(!showClusters)}
            variant={showClusters ? "default" : "outline"}
            size="sm"
          >
            {showClusters ? 'On' : 'Off'}
          </Button>
        </div>

        {/* Node Size Slider */}
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <Label className="text-sm font-medium whitespace-nowrap">
            Node Size:
          </Label>
          <Slider
            value={nodeSize}
            onValueChange={setNodeSize}
            min={1}
            max={10}
            step={1}
            className="flex-1"
          />
          <span className="text-sm text-muted-foreground w-8">
            {nodeSize[0]}
          </span>
        </div>

        {/* Display Options */}
        <div className="flex items-center gap-2">
          <Label className="text-sm font-medium whitespace-nowrap">
            Display:
          </Label>
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="sm"
              title="Show Labels"
            >
              🏷️
            </Button>
            <Button
              variant="outline"
              size="sm"
              title="Color by Attribute"
            >
              🎨
            </Button>
            <Button
              variant="outline"
              size="sm"
              title="Filter"
            >
              🔍
            </Button>
          </div>
        </div>

        {/* Simulation Button */}
        <Button className="ml-auto bg-green-600 hover:bg-green-700">
          ▶ Run Simulation
        </Button>
      </div>
    </div>
  )
}
