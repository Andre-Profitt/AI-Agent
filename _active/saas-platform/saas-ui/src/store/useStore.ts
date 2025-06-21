import { create } from 'zustand'
import { Message, Agent, Task } from '../types'

interface AppState {
  // Authentication
  isAuthenticated: boolean
  user: any | null
  token: string | null
  
  // Agents
  agents: Agent[]
  selectedAgent: Agent | null
  
  // Messages
  messages: Message[]
  
  // Tasks
  tasks: Task[]
  
  // UI State
  isLoading: boolean
  error: string | null
  sidebarOpen: boolean
  theme: 'light' | 'dark'
  
  // Actions
  setAuth: (isAuthenticated: boolean, user: any, token: string | null) => void
  selectAgent: (agent: Agent) => void
  addMessage: (message: Message) => void
  addTask: (task: Task) => void
  updateTask: (taskId: string, updates: Partial<Task>) => void
  setLoading: (isLoading: boolean) => void
  setError: (error: string | null) => void
  toggleSidebar: () => void
  toggleTheme: () => void
  reset: () => void
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  isAuthenticated: false,
  user: null,
  token: localStorage.getItem('token'),
  agents: [],
  selectedAgent: null,
  messages: [],
  tasks: [],
  isLoading: false,
  error: null,
  sidebarOpen: true,
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'dark',
  
  // Actions
  setAuth: (isAuthenticated, user, token) => {
    if (token) {
      localStorage.setItem('token', token)
    } else {
      localStorage.removeItem('token')
    }
    set({ isAuthenticated, user, token })
  },
  
  selectAgent: (agent) => set({ selectedAgent: agent }),
  
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),
  
  addTask: (task) => set((state) => ({
    tasks: [...state.tasks, task]
  })),
  
  updateTask: (taskId, updates) => set((state) => ({
    tasks: state.tasks.map((task) =>
      task.id === taskId ? { ...task, ...updates } : task
    )
  })),
  
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  
  toggleTheme: () => set((state) => {
    const newTheme = state.theme === 'light' ? 'dark' : 'light'
    localStorage.setItem('theme', newTheme)
    document.documentElement.classList.toggle('dark')
    return { theme: newTheme }
  }),
  
  reset: () => set({
    isAuthenticated: false,
    user: null,
    token: null,
    agents: [],
    selectedAgent: null,
    messages: [],
    tasks: [],
    isLoading: false,
    error: null,
  }),
}))