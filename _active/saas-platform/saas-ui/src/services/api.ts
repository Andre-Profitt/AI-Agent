import axios from 'axios'
import { Agent, Task, Message, HealthMetrics } from '../types'

const API_BASE_URL = '/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// API Service
export const apiService = {
  // Authentication
  login: async (credentials: { username: string; password: string }) => {
    const response = await api.post('/auth/login', credentials)
    return response.data
  },

  // Agents
  getAgents: async (): Promise<Agent[]> => {
    const response = await api.get('/agents')
    return response.data
  },

  getAgent: async (id: string): Promise<Agent> => {
    const response = await api.get(`/agents/${id}`)
    return response.data
  },

  createAgent: async (agent: Partial<Agent>): Promise<Agent> => {
    const response = await api.post('/agents', agent)
    return response.data
  },

  updateAgent: async (id: string, updates: Partial<Agent>): Promise<Agent> => {
    const response = await api.put(`/agents/${id}`, updates)
    return response.data
  },

  deleteAgent: async (id: string): Promise<void> => {
    await api.delete(`/agents/${id}`)
  },

  // Tasks
  getTasks: async (): Promise<Task[]> => {
    const response = await api.get('/tasks')
    return response.data
  },

  getTask: async (id: string): Promise<Task> => {
    const response = await api.get(`/tasks/${id}`)
    return response.data
  },

  createTask: async (task: {
    title: string
    description?: string
    agentId?: string
    priority?: 'low' | 'medium' | 'high'
  }): Promise<Task> => {
    const response = await api.post('/tasks', task)
    return response.data
  },

  updateTaskStatus: async (
    id: string,
    status: Task['status']
  ): Promise<Task> => {
    const response = await api.patch(`/tasks/${id}/status`, { status })
    return response.data
  },

  // Messages
  sendMessage: async (
    content: string,
    agentId?: string
  ): Promise<Message> => {
    const response = await api.post('/messages', { content, agentId })
    return response.data
  },

  getMessages: async (sessionId?: string): Promise<Message[]> => {
    const response = await api.get('/messages', {
      params: { sessionId },
    })
    return response.data
  },

  // Health & Metrics
  getHealth: async (): Promise<HealthMetrics> => {
    const response = await api.get('/health')
    return response.data
  },

  getMetrics: async () => {
    const response = await api.get('/metrics')
    return response.data
  },

  // Tools
  getTools: async () => {
    const response = await api.get('/tools')
    return response.data
  },

  // Sessions
  getSessions: async () => {
    const response = await api.get('/sessions')
    return response.data
  },

  getSession: async (id: string) => {
    const response = await api.get(`/sessions/${id}`)
    return response.data
  },
}

export default apiService