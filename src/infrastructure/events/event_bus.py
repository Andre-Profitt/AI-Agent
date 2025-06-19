"""
Event-driven architecture implementation for decoupled component communication.

Usage Example:
    from src.infrastructure.events.event_bus import get_event_bus, Event, EventType
    import asyncio
    
    async def my_handler(event: Event):
        logger.info("Handled event: {} from {}", extra={"event_type": event.type, "event_source": event.source})
    
    async def main():
        bus = get_event_bus()
        bus.subscribe(my_handler, event_types={EventType.SYSTEM_STARTUP})
        await bus.start()
        await bus.publish(Event(type=EventType.SYSTEM_STARTUP, source="example"))
        await asyncio.sleep(0.1)
        await bus.shutdown()
    
    asyncio.run(main())
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Awaitable, Union
from uuid import UUID, uuid4
import json
import traceback

logger = logging.getLogger(__name__)

# ... (EventType, Event, EventHandler, EventFilter, EventSubscription unchanged) ...

class EventBus:
    """
    Central event bus for publish-subscribe pattern.
    
    Features:
    - Async event handling
    - Priority-based subscription ordering
    - Event filtering
    - Error handling and retries
    - Performance monitoring
    - Context manager support
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.subscriptions: List[EventSubscription] = []
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._stats: Dict[str, Any] = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "processing_time_total": 0.0
        }
        self._event_history: List[Event] = []
        self._max_history_size: int = 1000
        self._processing_task: Optional[asyncio.Task] = None
    
    async def __aenter__(self) -> "EventBus":
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()
    
    async def start(self) -> None:
        """Start event processing."""
        self.processing = True
        if not self._processing_task or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_events())
        # Publish startup event
        await self.publish(Event(
            type=EventType.SYSTEM_STARTUP,
            source="event_bus",
            data={"message": "Event bus started"}
        ))
    
    async def shutdown(self) -> None:
        """Shutdown the event bus, ensuring all events are processed."""
        # Publish shutdown event
        await self.publish(Event(
            type=EventType.SYSTEM_SHUTDOWN,
            source="event_bus",
            data={"message": "Event bus stopping"}
        ))
        self.processing = False
        if self._processing_task:
            await self._processing_task
        await self.event_queue.join()
    
    async def stop(self) -> None:
        """Alias for shutdown (for backward compatibility)."""
        await self.shutdown()
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        if not self.processing and event.type != EventType.SYSTEM_STARTUP:
            logger.warning("Event published to stopped event bus")
        try:
            await self.event_queue.put(event)
            self._stats["events_published"] += 1
            self._add_to_history(event)
            logger.debug(
                f"Event published: {event.type.value}",
                extra={"event_id": str(event.id)}
            )
        except asyncio.QueueFull:
            logger.error(
                f"Event queue full, dropping event: {event.type.value}",
                extra={"event_id": str(event.id)}
            )
            self._stats["events_failed"] += 1
    
    def subscribe(
        self,
        handler: Callable[[Event], Union[Awaitable[None], None]],
        event_types: Optional[Set[EventType]] = None,
        filter: Optional[EventFilter] = None,
        priority: int = 0
    ) -> UUID:
        """
        Subscribe to events.
        Args:
            handler: Function to handle events
            event_types: Set of event types to subscribe to
            filter: Custom event filter
            priority: Subscription priority (higher = earlier execution)
        Returns:
            Subscription ID
        """
        if filter is None and event_types:
            filter = EventFilter(event_types=event_types)
        subscription = EventSubscription(
            handler=handler,
            filter=filter,
            priority=priority
        )
        # Insert subscription sorted by priority
        insert_index = 0
        for i, sub in enumerate(self.subscriptions):
            if sub.priority < priority:
                insert_index = i
                break
            insert_index = i + 1
        self.subscriptions.insert(insert_index, subscription)
        logger.info(
            f"Subscription added: {subscription.id}",
            extra={"handler": str(handler), "priority": priority}
        )
        return subscription.id
    
    def unsubscribe(self, subscription_id: UUID) -> bool:
        """Unsubscribe from events."""
        for i, sub in enumerate(self.subscriptions):
            if sub.id == subscription_id:
                self.subscriptions.pop(i)
                logger.info("Subscription removed: {}", extra={"subscription_id": subscription_id})
                return True
        return False
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self.processing or not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                start_time = time.time()
                await self._handle_event(event)
                processing_time = time.time() - start_time
                self._stats["events_processed"] += 1
                self._stats["processing_time_total"] += processing_time
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error processing event: {}\n{}", extra={"str_e_": str(e), "traceback_format_exc": traceback.format_exc()})
                self._stats["events_failed"] += 1
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        tasks = []
        for subscription in self.subscriptions:
            if subscription.filter.matches(event):
                task = asyncio.create_task(
                    self._safe_handle(subscription, event)
                )
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_handle(self, subscription: EventSubscription, event: Event) -> None:
        """Safely handle an event and log full traceback on error."""
        try:
            await subscription.handle_event(event)
        except Exception as e:
            logger.error("Handler error: %s\n%s", e, traceback.format_exc(), ,
                         extra={"event_id": str(event.id), "handler": str(subscription.handler)})
    
    def _add_to_history(self, event: Event) -> None:
        """Add event to history."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        avg_processing_time = 0.0
        if self._stats["events_processed"] > 0:
            avg_processing_time = (
                self._stats["processing_time_total"] /
                self._stats["events_processed"]
            )
        return {
            **self._stats,
            "avg_processing_time": avg_processing_time,
            "queue_size": self.event_queue.qsize(),
            "subscriptions_count": len(self.subscriptions),
            "history_size": len(self._event_history)
        }
    
    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[EventType] = None
    ) -> List[Event]:
        """Get recent events from history."""
        events = self._event_history[-limit:]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events
    
    def query_event_history(self, filter_fn: Optional[Callable[[Event], bool]] = None) -> List[Event]:
        """Query event history with an arbitrary filter function."""
        if filter_fn is None:
            return list(self._event_history)
        return [e for e in self._event_history if filter_fn(e)]
    
    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all active subscriptions and their filters/handlers."""
        return [
            {
                "id": str(sub.id),
                "handler": str(sub.handler),
                "priority": sub.priority,
                "active": sub.active,
                "filter": {
                    "event_types": [et.value for et in sub.filter.event_types],
                    "sources": list(sub.filter.sources),
                    "correlation_ids": [str(cid) for cid in sub.filter.correlation_ids],
                    "custom_filter": bool(sub.filter.custom_filter)
                }
            }
            for sub in self.subscriptions
        ]
    
    def serialize_event_history(self, limit: int = 100) -> str:
        """Serialize recent event history to JSON."""
        events = self.get_recent_events(limit=limit)
        return json.dumps([e.to_dict() for e in events], indent=2)

# ... (rest of the file unchanged) ... 