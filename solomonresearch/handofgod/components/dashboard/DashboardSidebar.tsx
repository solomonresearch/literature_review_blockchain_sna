'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Home, Network, Play, BarChart3, Library } from 'lucide-react'

const menuItems = [
  { name: 'Home', href: '/dashboard', icon: Home },
  { name: 'Networks', href: '/dashboard/networks', icon: Network },
  { name: 'Simulations', href: '/dashboard/simulations', icon: Play },
  { name: 'Analytics', href: '/dashboard/analytics', icon: BarChart3 },
  { name: 'Library', href: '/dashboard/library', icon: Library },
]

export default function DashboardSidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 bg-card border-r border-border p-4">
      <nav className="space-y-2">
        {menuItems.map((item) => {
          const isActive = pathname === item.href
          const Icon = item.icon
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary/10 text-primary font-semibold'
                  : 'hover:bg-accent text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.name}</span>
            </Link>
          )
        })}
      </nav>

      <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <h3 className="font-semibold text-sm mb-2 text-blue-900 dark:text-blue-100">
          Quick Tips
        </h3>
        <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1">
          <li>• Press Space to pan</li>
          <li>• Scroll to zoom</li>
          <li>• Click to select nodes</li>
          <li>• Double-click to center</li>
        </ul>
      </div>
    </aside>
  )
}
