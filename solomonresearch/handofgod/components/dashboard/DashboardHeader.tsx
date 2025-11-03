'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { signOut } from '@/lib/supabase/auth'

export default function DashboardHeader() {
  const router = useRouter()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const handleSignOut = async () => {
    await signOut()
    router.push('/auth/login')
    router.refresh()
  }

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/dashboard">
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 cursor-pointer">
              Hand of God
            </h1>
          </Link>

          <div className="hidden md:flex items-center gap-4">
            <input
              type="search"
              placeholder="Search networks..."
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white w-64"
            />
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            + New Network
          </button>

          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold"
            >
              U
            </button>

            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-2 z-50">
                <Link
                  href="/dashboard/settings"
                  className="block px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  Settings
                </Link>
                <Link
                  href="/dashboard/profile"
                  className="block px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  Profile
                </Link>
                <hr className="my-2 border-gray-200 dark:border-gray-700" />
                <button
                  onClick={handleSignOut}
                  className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 text-red-600 dark:text-red-400"
                >
                  Sign Out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
