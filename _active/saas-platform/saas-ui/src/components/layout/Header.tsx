import React from 'react'
import { FiBell, FiSearch, FiUser } from 'react-icons/fi'
import Button from '../ui/Button'
import Input from '../ui/Input'
import { useStore } from '@/store/useStore'

export default function Header() {
  const { user } = useStore()

  return (
    <header className="flex h-16 items-center justify-between border-b bg-card px-6">
      <div className="flex items-center flex-1">
        <div className="relative w-96">
          <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search agents, tasks, or messages..."
            className="pl-10"
          />
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon">
          <FiBell className="h-5 w-5" />
        </Button>
        
        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-sm font-medium">{user?.name || 'User'}</p>
            <p className="text-xs text-muted-foreground">{user?.email || 'user@example.com'}</p>
          </div>
          <Button variant="ghost" size="icon" className="rounded-full">
            <FiUser className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  )
}