import { Message, Task } from '../types'

class WebSocketService {
  private ws: WebSocket | null = null
  private callbacks: Map<string, (data: any) => void> = new Map()

  connect(clientId: string, token?: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return
    }

    // Use native WebSocket for simpler connection
    this.ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`)
    
    this.ws.onopen = () => {
      console.log('WebSocket connected')
      this.emit('connect', null)
    }
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected')
      this.emit('disconnect', null)
    }
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      this.emit('error', error)
    }
    
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'message') {
          this.emit('message', data.data)
        } else if (data.type === 'task_update') {
          this.emit('task_update', data.data)
        } else if (data.type === 'agent_status') {
          this.emit('agent_status', data.data)
        } else if (data.type === 'metrics_update') {
          this.emit('metrics_update', data.data)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  sendMessage(content: string, agentId?: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ content, agentId }))
    }
  }

  subscribeToAgent(agentId: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe_agent', agentId }))
    }
  }

  unsubscribeFromAgent(agentId: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'unsubscribe_agent', agentId }))
    }
  }

  on(event: string, callback: (data: any) => void) {
    this.callbacks.set(event, callback)
  }

  off(event: string) {
    this.callbacks.delete(event)
  }

  private emit(event: string, data: any) {
    const callback = this.callbacks.get(event)
    if (callback) {
      callback(data)
    }
  }
}

export default new WebSocketService()