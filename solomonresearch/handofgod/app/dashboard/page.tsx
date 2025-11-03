import NetworkCanvas from '@/components/network/NetworkCanvas'
import NetworkControls from '@/components/network/NetworkControls'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export default function DashboardPage() {
  return (
    <div className="flex flex-col h-full">
      {/* Main Network Visualization Area */}
      <div className="flex-1 overflow-hidden">
        <NetworkCanvas />
      </div>

      {/* Controls */}
      <NetworkControls />

      {/* Side Panel for Node/Edge Inspector */}
      <div className="hidden xl:block absolute right-4 top-20 w-80 max-h-[calc(100vh-200px)] overflow-y-auto">
        <Card>
          <CardHeader>
            <CardTitle>Inspector</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="text-sm font-semibold mb-2">
                Selection
              </h4>
              <p className="text-sm text-muted-foreground">
                Click on a node or edge to inspect its properties
              </p>
            </div>

            <div className="border-t pt-4">
              <h4 className="text-sm font-semibold mb-2">
                Quick Actions
              </h4>
              <div className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  📊 Calculate Metrics
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  🔮 Detect Communities
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  💾 Export Network
                </Button>
              </div>
            </div>

            <div className="border-t pt-4">
              <h4 className="text-sm font-semibold mb-2">
                Simulations
              </h4>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center">
                  <Label className="text-sm">Type:</Label>
                  <span className="font-medium">Intervention</span>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="rounds" className="text-sm">Rounds:</Label>
                  <Input
                    id="rounds"
                    type="number"
                    defaultValue="100"
                    className="h-8"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="repetitions" className="text-sm">Repetitions:</Label>
                  <Input
                    id="repetitions"
                    type="number"
                    defaultValue="10"
                    className="h-8"
                  />
                </div>
                <Button className="w-full bg-orange-600 hover:bg-orange-700">
                  Configure & Run
                </Button>
              </div>
            </div>

            <div className="border-t pt-4">
              <h4 className="text-sm font-semibold mb-2">
                Recent Activity
              </h4>
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex items-start gap-2">
                  <span>📊</span>
                  <div>
                    <p>Calculated betweenness centrality</p>
                    <p className="text-muted-foreground/60">2 min ago</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span>🎮</span>
                  <div>
                    <p>Ran diffusion simulation</p>
                    <p className="text-muted-foreground/60">5 min ago</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span>💾</span>
                  <div>
                    <p>Saved network layout</p>
                    <p className="text-muted-foreground/60">10 min ago</p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
