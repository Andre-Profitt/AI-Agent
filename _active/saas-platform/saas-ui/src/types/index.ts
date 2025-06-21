export interface Agent {
  id: string
  name: string
  type: string
  description?: string
  capabilities: string[]
  status: 'online' | 'offline' | 'busy'
  avatar?: string
  metrics?: {
    tasksCompleted: number
    successRate: number
    avgResponseTime: number
  }
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  agentId?: string
  metadata?: {
    tools?: string[]
    reasoning?: string
    confidence?: number
  }
}

export interface Task {
  id: string
  title: string
  description?: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  priority: 'low' | 'medium' | 'high'
  agentId?: string
  createdAt: Date
  updatedAt: Date
  result?: any
  error?: string
}

export interface Session {
  id: string
  userId: string
  startTime: Date
  endTime?: Date
  messages: Message[]
  context?: any
}

export interface Tool {
  id: string
  name: string
  description: string
  category: string
  parameters?: any
}

export interface HealthMetrics {
  status: 'healthy' | 'degraded' | 'unhealthy'
  uptime: number
  activeAgents: number
  tasksInProgress: number
  memoryUsage: number
  cpuUsage: number
  latency: number
}