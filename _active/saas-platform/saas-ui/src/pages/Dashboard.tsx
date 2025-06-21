import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  FiActivity, 
  FiUsers, 
  FiMessageSquare, 
  FiTrendingUp,
  FiCpu,
  FiHardDrive,
  FiWifi
} from 'react-icons/fi'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import apiService from '@/services/api'
import { HealthMetrics } from '@/types'
import Skeleton from '@/components/ui/Skeleton'

const mockData = [
  { time: '00:00', cpu: 20, memory: 45, tasks: 12 },
  { time: '04:00', cpu: 35, memory: 52, tasks: 18 },
  { time: '08:00', cpu: 65, memory: 68, tasks: 32 },
  { time: '12:00', cpu: 78, memory: 72, tasks: 45 },
  { time: '16:00', cpu: 82, memory: 75, tasks: 38 },
  { time: '20:00', cpu: 45, memory: 58, tasks: 22 },
  { time: '24:00', cpu: 25, memory: 48, tasks: 15 },
]

export default function Dashboard() {
  const [metrics, setMetrics] = useState<HealthMetrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadMetrics()
    const interval = setInterval(loadMetrics, 5000) // Update every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const loadMetrics = async () => {
    try {
      const data = await apiService.getHealth()
      setMetrics(data)
    } catch (error) {
      console.error('Failed to load metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  const stats = [
    { 
      title: 'Active Agents', 
      value: metrics?.activeAgents || 0, 
      icon: FiUsers,
      change: '+12%',
      color: 'text-blue-500'
    },
    { 
      title: 'Tasks Running', 
      value: metrics?.tasksInProgress || 0, 
      icon: FiActivity,
      change: '+5%',
      color: 'text-green-500'
    },
    { 
      title: 'Messages Today', 
      value: '1,234', 
      icon: FiMessageSquare,
      change: '+23%',
      color: 'text-purple-500'
    },
    { 
      title: 'Avg Response', 
      value: `${metrics?.latency || 0}ms`, 
      icon: FiTrendingUp,
      change: '-8%',
      color: 'text-orange-500'
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">Monitor your AI agents and system performance</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  {stat.title}
                </CardTitle>
                <stat.icon className={`h-4 w-4 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                {loading ? (
                  <Skeleton className="h-8 w-24" />
                ) : (
                  <>
                    <div className="text-2xl font-bold">{stat.value}</div>
                    <p className="text-xs text-muted-foreground">
                      <span className="text-green-500">{stat.change}</span> from last hour
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>System Performance</CardTitle>
            <CardDescription>CPU and Memory usage over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="cpu" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="CPU %"
                />
                <Line 
                  type="monotone" 
                  dataKey="memory" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="Memory %"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Task Throughput</CardTitle>
            <CardDescription>Number of tasks processed per hour</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={mockData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="tasks" 
                  stroke="#8b5cf6" 
                  fill="#8b5cf6" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* System Health */}
      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>Real-time system status and metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-lg ${
                metrics?.status === 'healthy' ? 'bg-green-500/10' : 'bg-red-500/10'
              }`}>
                <FiWifi className={`h-6 w-6 ${
                  metrics?.status === 'healthy' ? 'text-green-500' : 'text-red-500'
                }`} />
              </div>
              <div>
                <p className="text-sm font-medium">Status</p>
                <p className="text-2xl font-bold capitalize">
                  {metrics?.status || 'Unknown'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="p-3 rounded-lg bg-blue-500/10">
                <FiCpu className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <p className="text-sm font-medium">CPU Usage</p>
                <p className="text-2xl font-bold">{metrics?.cpuUsage || 0}%</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="p-3 rounded-lg bg-purple-500/10">
                <FiHardDrive className="h-6 w-6 text-purple-500" />
              </div>
              <div>
                <p className="text-sm font-medium">Memory Usage</p>
                <p className="text-2xl font-bold">{metrics?.memoryUsage || 0}%</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}