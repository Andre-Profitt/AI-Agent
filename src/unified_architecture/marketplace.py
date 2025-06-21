from agent import query
from examples.parallel_execution_example import results
from performance_dashboard import stats
from setup_environment import value
from tests.load_test import data

from src.adapters.fsm_unified_adapter import capabilities
from src.collaboration.realtime_collaboration import updates
from src.core.monitoring import key
from src.database.models import agent_id
from src.database.models import status
from src.database.models import storage_backend
from src.database.models import user_id
from src.database.supabase_manager import existing
from src.meta_cognition import query_lower
from src.services.integration_hub import limit
from src.unified_architecture.marketplace import capability_set
from src.unified_architecture.marketplace import category_counts
from src.unified_architecture.marketplace import favorites
from src.unified_architecture.marketplace import listing
from src.unified_architecture.marketplace import listings
from src.unified_architecture.marketplace import listings_data
from src.unified_architecture.marketplace import ratings
from src.unified_architecture.marketplace import recent_listings
from src.unified_architecture.marketplace import tag_set
from src.unified_architecture.marketplace import total_rating
from src.unified_architecture.shared_memory import imported_count
from src.utils.tools_introspection import field

from src.agents.advanced_agent_fsm import AgentCapability

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Dict, Enum, List, Optional, Set, Tuple, UUID, agent_id, author, cap, capabilities, capability, capability_set, category_counts, data, dataclass, datetime, e, existing, favorites, field, imported_count, json, key, l, limit, listing, listing_data, listing_id, listings, listings_data, logging, min_rating, offset, query, query_lower, r, rating, ratings, recent_listings, results, stats, status, storage_backend, tag, tag_set, tags, total_rating, updates, user_id, uuid4, value, x
from src.core.entities.agent import AgentCapability


"""
import datetime
from datetime import datetime
from src.unified_architecture.enhanced_platform import AgentCapability
from uuid import uuid4
# TODO: Fix undefined variables: agent_id, author, cap, capabilities, capability, capability_set, category_counts, data, e, existing, favorites, imported_count, key, l, limit, listing, listing_data, listing_id, listings, listings_data, min_rating, offset, query, query_lower, r, rating, ratings, recent_listings, results, self, stats, status, storage_backend, tag, tag_set, tags, total_rating, updates, user_id, value, x

from fastapi import status
Agent Marketplace for Unified Architecture

This module provides a marketplace system for agent discovery, listing, rating,
and deployment in the multi-agent collaboration platform.
"""

from typing import Set
from typing import Tuple
from typing import Optional
from dataclasses import field
from typing import Any
from typing import List

import json
import logging

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from uuid import UUID, uuid4

class ListingStatus(Enum):
    """Status of an agent listing in the marketplace"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_REVIEW = "pending_review"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"

class RatingCategory(Enum):
    """Categories for agent ratings"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    COLLABORATION = "collaboration"
    DOCUMENTATION = "documentation"
    SUPPORT = "support"

@dataclass
class AgentRating:
    """Rating for an agent"""
    agent_id: UUID = field()
    reviewer_id: UUID = field()
    category: RatingCategory = field()
    score: float = field()  # 1.0 to 5.0
    id: UUID = field(default_factory=uuid4)
    review: str = field(default="")
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not 1.0 <= self.score <= 5.0:
            raise ValueError("Rating score must be between 1.0 and 5.0")

@dataclass
class AgentListing:
    """Listing for an agent in the marketplace"""
    agent_id: UUID = field()
    name: str = field()
    description: str = field()
    version: str = field()
    author: str = field()
    id: UUID = field(default_factory=uuid4)
    capabilities: List[AgentCapability] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: ListingStatus = field(default=ListingStatus.PENDING_REVIEW)
    pricing_model: str = field(default="free")
    pricing_details: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    documentation_url: Optional[str] = field(default=None)
    source_code_url: Optional[str] = field(default=None)
    license: str = field(default="MIT")
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    download_count: int = field(default=0)
    rating_count: int = field(default=0)
    average_rating: float = field(default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert listing to dictionary"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "capabilities": [cap.value for cap in self.capabilities],
            "tags": self.tags,
            "status": self.status.value,
            "pricing_model": self.pricing_model,
            "pricing_details": self.pricing_details,
            "requirements": self.requirements,
            "documentation_url": self.documentation_url,
            "source_code_url": self.source_code_url,
            "license": self.license,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "download_count": self.download_count,
            "rating_count": self.rating_count,
            "average_rating": self.average_rating
        }

@dataclass
class MarketplaceStats:
    """Marketplace statistics"""
    total_listings: int = 0
    active_listings: int = 0
    total_downloads: int = 0
    total_ratings: int = 0
    average_rating: float = 0.0
    top_categories: List[Tuple[str, int]] = field(default_factory=list)
    recent_activity: List[Dict[str, Any]] = field(default_factory=list)

class AgentMarketplace:
    """
    Marketplace for agent discovery, listing, and deployment
    """

    def __init__(self, storage_backend: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.storage_backend = storage_backend or "memory"

        # In-memory storage (can be replaced with database)
        self._listings: Dict[UUID, AgentListing] = {}
        self._ratings: Dict[UUID, List[AgentRating]] = {}
        self._downloads: Dict[UUID, int] = {}
        self._favorites: Dict[UUID, Set[UUID]] = {}  # user_id -> set of listing_ids

        # Search index
        self._capability_index: Dict[AgentCapability, Set[UUID]] = {}
        self._tag_index: Dict[str, Set[UUID]] = {}
        self._author_index: Dict[str, Set[UUID]] = {}

        self.logger.info("AgentMarketplace initialized with {} backend", extra={"self_storage_backend": self.storage_backend})

    async def create_listing(self, listing: AgentListing) -> AgentListing:
        """Create a new agent listing"""
        try:
            # Validate listing
            if not listing.name or not listing.description:
                raise ValueError("Name and description are required")

            # Check if agent already has a listing
            existing = await self.get_listing_by_agent_id(listing.agent_id)
            if existing:
                raise ValueError(f"Agent {listing.agent_id} already has a listing")

            # Set initial status
            listing.status = ListingStatus.PENDING_REVIEW
            listing.created_at = datetime.utcnow()
            listing.updated_at = datetime.utcnow()

            # Store listing
            self._listings[listing.id] = listing

            # Update indices
            await self._update_indices(listing)

            self.logger.info("Created listing {} for agent {}", extra={"listing_id": listing.id, "listing_agent_id": listing.agent_id})
            return listing

        except Exception as e:
            self.logger.error("Failed to create listing: {}", extra={"e": e})
            raise

    async def update_listing(self, listing_id: UUID, updates: Dict[str, Any]) -> AgentListing:
        """Update an existing listing"""
        try:
            if listing_id not in self._listings:
                raise ValueError(f"Listing {listing_id} not found")

            listing = self._listings[listing_id]

            # Update fields
            for key, value in updates.items():
                if hasattr(listing, key):
                    setattr(listing, key, value)

            listing.updated_at = datetime.utcnow()

            # Update indices
            await self._update_indices(listing)

            self.logger.info("Updated listing {}", extra={"listing_id": listing_id})
            return listing

        except Exception as e:
            self.logger.error("Failed to update listing {}: {}", extra={"listing_id": listing_id, "e": e})
            raise

    async def get_listing(self, listing_id: UUID) -> Optional[AgentListing]:
        """Get a listing by ID"""
        return self._listings.get(listing_id)

    async def get_listing_by_agent_id(self, agent_id: UUID) -> Optional[AgentListing]:
        """Get a listing by agent ID"""
        for listing in self._listings.values():
            if listing.agent_id == agent_id:
                return listing
        return None

    async def search_listings(
        self,
        query: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        status: Optional[ListingStatus] = None,
        min_rating: Optional[float] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[AgentListing]:
        """Search listings with various filters"""
        try:
            results = list(self._listings.values())

            # Filter by status
            if status:
                results = [r for r in results if r.status == status]

            # Filter by capabilities
            if capabilities:
                capability_set = set(capabilities)
                results = [r for r in results if capability_set.issubset(set(r.capabilities))]

            # Filter by tags
            if tags:
                tag_set = set(tags)
                results = [r for r in results if tag_set.issubset(set(r.tags))]

            # Filter by author
            if author:
                results = [r for r in results if author.lower() in r.author.lower()]

            # Filter by minimum rating
            if min_rating:
                results = [r for r in results if r.average_rating >= min_rating]

            # Text search
            if query:
                query_lower = query.lower()
                results = [
                    r for r in results
                    if (query_lower in r.name.lower() or
                        query_lower in r.description.lower() or
                        any(query_lower in tag.lower() for tag in r.tags))
                ]

            # Sort by rating and download count
            results.sort(key=lambda x: (x.average_rating, x.download_count), reverse=True)

            # Apply pagination
            return results[offset:offset + limit]

        except Exception as e:
            self.logger.error("Search failed: {}", extra={"e": e})
            return []

    async def add_rating(self, rating: AgentRating) -> AgentRating:
        """Add a rating to an agent"""
        try:
            # Validate listing exists
            listing = await self.get_listing_by_agent_id(rating.agent_id)
            if not listing:
                raise ValueError(f"Agent {rating.agent_id} not found in marketplace")

            # Store rating
            if rating.agent_id not in self._ratings:
                self._ratings[rating.agent_id] = []

            self._ratings[rating.agent_id].append(rating)

            # Update listing statistics
            await self._update_listing_stats(listing)

            self.logger.info("Added rating {} for agent {}", extra={"rating_id": rating.id, "rating_agent_id": rating.agent_id})
            return rating

        except Exception as e:
            self.logger.error("Failed to add rating: {}", extra={"e": e})
            raise

    async def get_ratings(self, agent_id: UUID) -> List[AgentRating]:
        """Get all ratings for an agent"""
        return self._ratings.get(agent_id, [])

    async def download_agent(self, listing_id: UUID, user_id: UUID) -> bool:
        """Record an agent download"""
        try:
            listing = await self.get_listing(listing_id)
            if not listing:
                raise ValueError(f"Listing {listing_id} not found")

            # Increment download count
            listing.download_count += 1
            listing.updated_at = datetime.utcnow()

            # Track user download
            if listing_id not in self._downloads:
                self._downloads[listing_id] = 0
            self._downloads[listing_id] += 1

            self.logger.info("Agent {} downloaded by user {}", extra={"listing_agent_id": listing.agent_id, "user_id": user_id})
            return True

        except Exception as e:
            self.logger.error("Failed to record download: {}", extra={"e": e})
            return False

    async def add_to_favorites(self, user_id: UUID, listing_id: UUID) -> bool:
        """Add a listing to user's favorites"""
        try:
            if listing_id not in self._listings:
                raise ValueError(f"Listing {listing_id} not found")

            if user_id not in self._favorites:
                self._favorites[user_id] = set()

            self._favorites[user_id].add(listing_id)
            return True

        except Exception as e:
            self.logger.error("Failed to add to favorites: {}", extra={"e": e})
            return False

    async def remove_from_favorites(self, user_id: UUID, listing_id: UUID) -> bool:
        """Remove a listing from user's favorites"""
        try:
            if user_id in self._favorites:
                self._favorites[user_id].discard(listing_id)
            return True

        except Exception as e:
            self.logger.error("Failed to remove from favorites: {}", extra={"e": e})
            return False

    async def get_favorites(self, user_id: UUID) -> List[AgentListing]:
        """Get user's favorite listings"""
        try:
            if user_id not in self._favorites:
                return []

            favorites = []
            for listing_id in self._favorites[user_id]:
                listing = await self.get_listing(listing_id)
                if listing:
                    favorites.append(listing)

            return favorites

        except Exception as e:
            self.logger.error("Failed to get favorites: {}", extra={"e": e})
            return []

    async def get_marketplace_stats(self) -> MarketplaceStats:
        """Get marketplace statistics"""
        try:
            stats = MarketplaceStats()

            # Basic counts
            stats.total_listings = len(self._listings)
            stats.active_listings = len([l for l in self._listings.values()
                                       if l.status == ListingStatus.ACTIVE])
            stats.total_downloads = sum(self._downloads.values())
            stats.total_ratings = sum(len(ratings) for ratings in self._ratings.values())

            # Average rating
            if stats.total_ratings > 0:
                total_rating = sum(
                    sum(r.score for r in ratings)
                    for ratings in self._ratings.values()
                )
                stats.average_rating = total_rating / stats.total_ratings

            # Top categories
            category_counts = {}
            for listing in self._listings.values():
                for capability in listing.capabilities:
                    category_counts[capability.value] = category_counts.get(capability.value, 0) + 1

            stats.top_categories = sorted(
                category_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Recent activity
            recent_listings = sorted(
                self._listings.values(),
                key=lambda x: x.updated_at,
                reverse=True
            )[:10]

            stats.recent_activity = [
                {
                    "type": "listing_update",
                    "listing_id": str(l.id),
                    "agent_name": l.name,
                    "timestamp": l.updated_at.isoformat()
                }
                for l in recent_listings
            ]

            return stats

        except Exception as e:
            self.logger.error("Failed to get marketplace stats: {}", extra={"e": e})
            return MarketplaceStats()

    async def _update_indices(self, listing: AgentListing):
        """Update search indices for a listing"""
        try:
            # Capability index
            for capability in listing.capabilities:
                if capability not in self._capability_index:
                    self._capability_index[capability] = set()
                self._capability_index[capability].add(listing.id)

            # Tag index
            for tag in listing.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(listing.id)

            # Author index
            if listing.author not in self._author_index:
                self._author_index[listing.author] = set()
            self._author_index[listing.author].add(listing.id)

        except Exception as e:
            self.logger.error("Failed to update indices: {}", extra={"e": e})

    async def _update_listing_stats(self, listing: AgentListing):
        """Update listing statistics from ratings"""
        try:
            ratings = await self.get_ratings(listing.agent_id)
            if ratings:
                listing.rating_count = len(ratings)
                listing.average_rating = sum(r.score for r in ratings) / len(ratings)
                listing.updated_at = datetime.utcnow()

        except Exception as e:
            self.logger.error("Failed to update listing stats: {}", extra={"e": e})

    async def export_listings(self, format: str = "json") -> str:
        """Export all listings"""
        try:
            listings = [listing.to_dict() for listing in self._listings.values()]

            if format.lower() == "json":
                return json.dumps(listings, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error("Failed to export listings: {}", extra={"e": e})
            return ""

    async def import_listings(self, data: str, format: str = "json") -> int:
        """Import listings from external source"""
        try:
            if format.lower() == "json":
                listings_data = json.loads(data)
            else:
                raise ValueError(f"Unsupported import format: {format}")

            imported_count = 0
            for listing_data in listings_data:
                try:
                    # Convert string IDs back to UUIDs
                    listing_data["id"] = UUID(listing_data["id"])
                    listing_data["agent_id"] = UUID(listing_data["agent_id"])

                    # Convert capabilities back to enums
                    listing_data["capabilities"] = [
                        AgentCapability(cap) for cap in listing_data["capabilities"]
                    ]

                    # Convert status back to enum
                    listing_data["status"] = ListingStatus(listing_data["status"])

                    # Create listing
                    listing = AgentListing(**listing_data)
                    await self.create_listing(listing)
                    imported_count += 1

                except Exception as e:
                    self.logger.warning("Failed to import listing: {}", extra={"e": e})
                    continue

            self.logger.info("Imported {} listings", extra={"imported_count": imported_count})
            return imported_count

        except Exception as e:
            self.logger.error("Failed to import listings: {}", extra={"e": e})
            return 0