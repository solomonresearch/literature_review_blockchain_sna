'use client'

import { useRef, useEffect, useState } from 'react'

interface Node {
  id: string
  x: number
  y: number
  radius: number
  color: string
}

interface Edge {
  source: string
  target: string
}

export default function NetworkCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })

  // Generate mock network data
  useEffect(() => {
    const mockNodes: Node[] = []
    const mockEdges: Edge[] = []
    const nodeCount = 100
    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']

    // Generate random nodes
    for (let i = 0; i < nodeCount; i++) {
      mockNodes.push({
        id: `node-${i}`,
        x: Math.random() * 800 - 400,
        y: Math.random() * 600 - 300,
        radius: Math.random() * 8 + 4,
        color: colors[Math.floor(Math.random() * colors.length)],
      })
    }

    // Generate random edges
    for (let i = 0; i < nodeCount * 2; i++) {
      const source = mockNodes[Math.floor(Math.random() * nodeCount)]
      const target = mockNodes[Math.floor(Math.random() * nodeCount)]
      if (source.id !== target.id) {
        mockEdges.push({
          source: source.id,
          target: target.id,
        })
      }
    }

    setNodes(mockNodes)
    setEdges(mockEdges)
  }, [])

  // Draw network
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Apply transformations
      ctx.save()
      ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y)
      ctx.scale(zoom, zoom)

      // Draw edges
      ctx.strokeStyle = 'rgba(156, 163, 175, 0.3)'
      ctx.lineWidth = 0.5
      edges.forEach((edge) => {
        const sourceNode = nodes.find((n) => n.id === edge.source)
        const targetNode = nodes.find((n) => n.id === edge.target)
        if (sourceNode && targetNode) {
          ctx.beginPath()
          ctx.moveTo(sourceNode.x, sourceNode.y)
          ctx.lineTo(targetNode.x, targetNode.y)
          ctx.stroke()
        }
      })

      // Draw nodes
      nodes.forEach((node) => {
        ctx.fillStyle = node.color
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        ctx.fill()

        // Draw node border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
        ctx.lineWidth = 1
        ctx.stroke()
      })

      ctx.restore()
    }

    draw()
  }, [nodes, edges, zoom, pan])

  // Handle mouse wheel for zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setZoom((prev) => Math.max(0.1, Math.min(5, prev * delta)))
  }

  return (
    <div className="relative w-full h-full bg-gray-50 dark:bg-gray-900">
      <canvas
        ref={canvasRef}
        width={1200}
        height={800}
        onWheel={handleWheel}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {/* Overlay stats */}
      <div className="absolute top-4 left-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 min-w-[200px]">
        <h3 className="font-semibold mb-2 text-sm">Network Stats</h3>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Nodes:</span>
            <span className="font-semibold">{nodes.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Edges:</span>
            <span className="font-semibold">{edges.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Density:</span>
            <span className="font-semibold">0.024</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Zoom:</span>
            <span className="font-semibold">{zoom.toFixed(2)}x</span>
          </div>
        </div>
      </div>

      {/* Controls overlay */}
      <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-2 flex gap-2">
        <button
          onClick={() => setZoom((prev) => Math.min(5, prev * 1.2))}
          className="px-3 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
          title="Zoom In"
        >
          +
        </button>
        <button
          onClick={() => setZoom((prev) => Math.max(0.1, prev * 0.8))}
          className="px-3 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
          title="Zoom Out"
        >
          −
        </button>
        <button
          onClick={() => {
            setZoom(1)
            setPan({ x: 0, y: 0 })
          }}
          className="px-3 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
          title="Reset View"
        >
          ⟲
        </button>
      </div>
    </div>
  )
}
