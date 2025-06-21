import React, { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Chat from './pages/Chat'
import Agents from './pages/Agents'
import { useStore } from './store/useStore'

// Placeholder pages
const Tools = () => <div className="text-2xl">Tools Page (Coming Soon)</div>
const Monitoring = () => <div className="text-2xl">Monitoring Page (Coming Soon)</div>
const Settings = () => <div className="text-2xl">Settings Page (Coming Soon)</div>

function App() {
  const { theme } = useStore()

  useEffect(() => {
    // Apply theme on mount
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  return (
    <Router>
      <div className="min-h-screen bg-background text-foreground">
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="chat" element={<Chat />} />
            <Route path="agents" element={<Agents />} />
            <Route path="tools" element={<Tools />} />
            <Route path="monitoring" element={<Monitoring />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
        <Toaster
          position="bottom-right"
          toastOptions={{
            className: '',
            style: {
              background: 'hsl(var(--background))',
              color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))',
            },
          }}
        />
      </div>
    </Router>
  )
}

export default App