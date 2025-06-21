import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { FiPlus, FiSearch, FiFilter } from 'react-icons/fi'
import { useStore } from '@/store/useStore'
import AgentCard from '@/components/agents/AgentCard'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import apiService from '@/services/api'
import Skeleton from '@/components/ui/Skeleton'

export default function Agents() {
  const { agents } = useStore()
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState('all')

  useEffect(() => {
    loadAgents()
  }, [])

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

  const filteredAgents = agents.filter((agent) => {
    const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      agent.description?.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = filterType === 'all' || agent.type === filterType
    return matchesSearch && matchesType
  })

  const agentTypes = ['all', ...new Set(agents.map(a => a.type))]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Agents</h1>
          <p className="text-muted-foreground">Manage and configure your AI agents</p>
        </div>
        <Button>
          <FiPlus className="mr-2" />
          Create Agent
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex gap-4">
        <div className="relative flex-1">
          <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search agents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <div className="flex gap-2">
          {agentTypes.map((type) => (
            <Button
              key={type}
              variant={filterType === type ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterType(type)}
            >
              {type === 'all' ? 'All' : type}
            </Button>
          ))}
        </div>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-64" />
          ))
        ) : filteredAgents.length > 0 ? (
          filteredAgents.map((agent, index) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <AgentCard
                agent={agent}
                onViewDetails={(agent) => {
                  // TODO: Show agent details modal
                  console.log('View details:', agent)
                }}
              />
            </motion.div>
          ))
        ) : (
          <div className="col-span-full text-center py-12">
            <p className="text-muted-foreground">
              {searchTerm || filterType !== 'all'
                ? 'No agents found matching your criteria'
                : 'No agents available'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}