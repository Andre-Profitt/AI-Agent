import React from 'react'
import { motion } from 'framer-motion'
import { 
  FiHome, 
  FiMessageSquare, 
  FiUsers, 
  FiSettings,
  FiActivity,
  FiTool,
  FiChevronLeft,
  FiChevronRight,
  FiMoon,
  FiSun
} from 'react-icons/fi'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { useStore } from '@/store/useStore'
import Button from '../ui/Button'

const navigation = [
  { name: 'Dashboard', href: '/', icon: FiHome },
  { name: 'Chat', href: '/chat', icon: FiMessageSquare },
  { name: 'Agents', href: '/agents', icon: FiUsers },
  { name: 'Tools', href: '/tools', icon: FiTool },
  { name: 'Monitoring', href: '/monitoring', icon: FiActivity },
  { name: 'Settings', href: '/settings', icon: FiSettings },
]

export default function Sidebar() {
  const location = useLocation()
  const { sidebarOpen, toggleSidebar, theme, toggleTheme } = useStore()

  return (
    <motion.div
      initial={{ width: sidebarOpen ? 240 : 60 }}
      animate={{ width: sidebarOpen ? 240 : 60 }}
      transition={{ duration: 0.2 }}
      className="flex h-full flex-col bg-card border-r"
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between px-4">
        <motion.div
          initial={{ opacity: sidebarOpen ? 1 : 0 }}
          animate={{ opacity: sidebarOpen ? 1 : 0 }}
          className="flex items-center"
        >
          <span className="text-xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
            AI Agent Hub
          </span>
        </motion.div>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className="ml-auto"
        >
          {sidebarOpen ? <FiChevronLeft /> : <FiChevronRight />}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center rounded-md px-2 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {sidebarOpen && (
                <span className="ml-3">{item.name}</span>
              )}
            </Link>
          )
        })}
      </nav>

      {/* Theme Toggle */}
      <div className="border-t p-4">
        <Button
          variant="ghost"
          size={sidebarOpen ? 'md' : 'icon'}
          onClick={toggleTheme}
          className="w-full justify-start"
        >
          {theme === 'dark' ? (
            <FiSun className="h-5 w-5" />
          ) : (
            <FiMoon className="h-5 w-5" />
          )}
          {sidebarOpen && <span className="ml-3">Toggle Theme</span>}
        </Button>
      </div>
    </motion.div>
  )
}