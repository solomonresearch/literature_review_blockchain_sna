import LoginForm from '@/components/auth/LoginForm'
import Link from 'next/link'

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link href="/">
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 cursor-pointer">
              Hand of God
            </h1>
          </Link>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Network Visualization Platform
          </p>
        </div>

        <LoginForm />
      </div>
    </div>
  )
}
