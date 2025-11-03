import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import '../styles/globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Hand of God - Network Visualization Platform',
  description: 'High-performance network visualization and simulation platform for large-scale network analysis',
  keywords: ['network', 'visualization', 'simulation', 'graph', 'analysis', 'gephi'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
