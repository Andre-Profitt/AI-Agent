import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import ChatInterface from '@/components/chat/ChatInterface'
import AgentCard from '@/components/agents/AgentCard'
import { useStore } from '@/store/useStore'
import apiService from '@/services/api'
import wsService from '@/services/websocket'
import Skeleton from '@/components/ui/Skeleton'

export default function Chat() {
  const { agents, selectedAgent, selectAgent } = useStore()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadAgents()
    // Connect WebSocket
    wsService.connect('web-client-' + Date.now())
    
    return () => {
      wsService.disconnect()
    }
  }, [])

  useEffect(() => {
    if (selectedAgent) {
      wsService.subscribeToAgent(selectedAgent.id)
    }
  }, [selectedAgent])

  const loadAgents = async () => {
    try {
      const agentsData = await apiService.getAgents()
      useStore.setState({ agents: agentsData })
    } catch (error) {
      console.error('Failed to load agents:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex gap-6 h-full">
      {/* Agent Selection */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="w-80 space-y-4"
      >
        <div>
          <h2 className="text-xl font-semibold mb-2">Select an Agent</h2>
          <p className="text-sm text-muted-foreground">
            Choose an AI agent to start chatting
          </p>
        </div>

        <div className="space-y-3">
          {loading ? (
            <>
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
            </>
          ) : agents.length > 0 ? (
            agents.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                isSelected={selectedAgent?.id === agent.id}
                onSelect={selectAgent}
              />
            ))
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No agents available
            </div>
          )}
        </div>
      </motion.div>

      {/* Chat Interface */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex-1"
      >
        <ChatInterface />
      </motion.div>
    </div>
  )
}