'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { signOut } from '@/lib/supabase/auth'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export default function DashboardHeader() {
  const router = useRouter()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const handleSignOut = async () => {
    await signOut()
    router.push('/auth/login')
    router.refresh()
  }

  return (
    <header className="bg-card border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/dashboard">
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 cursor-pointer">
              Hand of God
            </h1>
          </Link>

          <div className="hidden md:flex items-center gap-4">
            <Input
              type="search"
              placeholder="Search networks..."
              className="w-64"
            />
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Button>+ New Network</Button>

          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold hover:opacity-90 transition-opacity"
            >
              U
            </button>

            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-card rounded-lg shadow-lg border border-border py-2 z-50">
                <Link
                  href="/dashboard/settings"
                  className="block px-4 py-2 hover:bg-accent"
                >
                  Settings
                </Link>
                <Link
                  href="/dashboard/profile"
                  className="block px-4 py-2 hover:bg-accent"
                >
                  Profile
                </Link>
                <hr className="my-2 border-border" />
                <button
                  onClick={handleSignOut}
                  className="w-full text-left px-4 py-2 hover:bg-accent text-destructive"
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
