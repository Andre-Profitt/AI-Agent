import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiSend, FiPaperclip, FiMic, FiStopCircle } from 'react-icons/fi'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useStore } from '@/store/useStore'
import Button from '../ui/Button'
import Input from '../ui/Input'
import ScrollArea from '../ui/ScrollArea'
import { cn } from '@/lib/utils'
import apiService from '@/services/api'
import wsService from '@/services/websocket'

export default function ChatInterface() {
  const { messages, selectedAgent, addMessage, isLoading, setLoading } = useStore()
  const [input, setInput] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    // Subscribe to WebSocket messages
    wsService.on('message', (message) => {
      addMessage(message)
      setLoading(false)
    })

    return () => {
      wsService.off('message')
    }
  }, [addMessage, setLoading])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = {
      id: Date.now().toString(),
      role: 'user' as const,
      content: input,
      timestamp: new Date(),
      agentId: selectedAgent?.id,
    }

    addMessage(userMessage)
    setInput('')
    setLoading(true)

    try {
      // Send via WebSocket for real-time response
      wsService.sendMessage(input, selectedAgent?.id)
    } catch (error) {
      console.error('Failed to send message:', error)
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const toggleRecording = () => {
    setIsRecording(!isRecording)
    // TODO: Implement voice recording
  }

  return (
    <div className="flex flex-col h-full bg-background rounded-lg border">
      {/* Messages */}
      <ScrollArea
        ref={scrollAreaRef}
        className="flex-1 p-4 space-y-4"
      >
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={cn(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={cn(
                  'max-w-[70%] rounded-lg px-4 py-2',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                )}
              >
                {message.role === 'assistant' && selectedAgent && (
                  <div className="flex items-center gap-2 mb-2 text-sm text-muted-foreground">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    {selectedAgent.name}
                  </div>
                )}
                
                <ReactMarkdown
                  className="prose prose-sm dark:prose-invert max-w-none"
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '')
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      )
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>

                {message.metadata?.tools && (
                  <div className="mt-2 text-xs text-muted-foreground">
                    Tools: {message.metadata.tools.join(', ')}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-muted rounded-lg px-4 py-2">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-foreground/50 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </motion.div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon">
            <FiPaperclip className="h-5 w-5" />
          </Button>
          
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              selectedAgent
                ? `Ask ${selectedAgent.name} anything...`
                : 'Select an agent to start chatting...'
            }
            disabled={!selectedAgent || isLoading}
            className="flex-1"
          />
          
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleRecording}
            className={cn(isRecording && 'text-red-500')}
          >
            {isRecording ? (
              <FiStopCircle className="h-5 w-5" />
            ) : (
              <FiMic className="h-5 w-5" />
            )}
          </Button>
          
          <Button
            onClick={handleSend}
            disabled={!input.trim() || !selectedAgent || isLoading}
          >
            <FiSend className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}