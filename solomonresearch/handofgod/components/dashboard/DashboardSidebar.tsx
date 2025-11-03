'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const menuItems = [
  { name: 'Home', href: '/dashboard', icon: '🏠' },
  { name: 'Networks', href: '/dashboard/networks', icon: '🕸️' },
  { name: 'Simulations', href: '/dashboard/simulations', icon: '🎮' },
  { name: 'Analytics', href: '/dashboard/analytics', icon: '📊' },
  { name: 'Library', href: '/dashboard/library', icon: '📚' },
]

export default function DashboardSidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 p-4">
      <nav className="space-y-2">
        {menuItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 font-semibold'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              <span className="text-2xl">{item.icon}</span>
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
