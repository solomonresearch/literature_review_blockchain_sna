import DashboardHeader from '@/components/dashboard/DashboardHeader'
import DashboardSidebar from '@/components/dashboard/DashboardSidebar'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col h-screen">
      <DashboardHeader />
      <div className="flex flex-1 overflow-hidden">
        <DashboardSidebar />
        <main className="flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  )
}
