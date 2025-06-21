import React from 'react'
import { motion } from 'framer-motion'
import { FiActivity, FiCheckCircle, FiClock } from 'react-icons/fi'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card'
import Button from '../ui/Button'
import { Agent } from '@/types'
import { cn } from '@/lib/utils'

interface AgentCardProps {
  agent: Agent
  isSelected?: boolean
  onSelect?: (agent: Agent) => void
  onViewDetails?: (agent: Agent) => void
}

export default function AgentCard({ 
  agent, 
  isSelected, 
  onSelect, 
  onViewDetails 
}: AgentCardProps) {
  const statusColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-500',
    busy: 'bg-yellow-500',
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Card
        className={cn(
          'cursor-pointer transition-all',
          isSelected && 'ring-2 ring-primary'
        )}
        onClick={() => onSelect?.(agent)}
      >
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary to-primary/50 flex items-center justify-center text-white font-bold">
                  {agent.name.charAt(0).toUpperCase()}
                </div>
                <div
                  className={cn(
                    'absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-background',
                    statusColors[agent.status]
                  )}
                />
              </div>
              <div>
                <CardTitle className="text-lg">{agent.name}</CardTitle>
                <CardDescription>{agent.type}</CardDescription>
              </div>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            {agent.description || 'A powerful AI agent ready to help'}
          </p>
          
          {agent.capabilities && (
            <div className="flex flex-wrap gap-2 mb-4">
              {agent.capabilities.slice(0, 3).map((capability) => (
                <span
                  key={capability}
                  className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full"
                >
                  {capability}
                </span>
              ))}
              {agent.capabilities.length > 3 && (
                <span className="text-xs text-muted-foreground">
                  +{agent.capabilities.length - 3} more
                </span>
              )}
            </div>
          )}
          
          {agent.metrics && (
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="flex items-center gap-1">
                <FiCheckCircle className="text-green-500" />
                <span>{agent.metrics.tasksCompleted} tasks</span>
              </div>
              <div className="flex items-center gap-1">
                <FiActivity className="text-blue-500" />
                <span>{agent.metrics.successRate}% success</span>
              </div>
              <div className="flex items-center gap-1">
                <FiClock className="text-orange-500" />
                <span>{agent.metrics.avgResponseTime}ms</span>
              </div>
            </div>
          )}
          
          <div className="mt-4 flex gap-2">
            <Button
              size="sm"
              variant={isSelected ? 'default' : 'outline'}
              className="flex-1"
              onClick={(e) => {
                e.stopPropagation()
                onSelect?.(agent)
              }}
            >
              {isSelected ? 'Selected' : 'Select'}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation()
                onViewDetails?.(agent)
              }}
            >
              Details
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}