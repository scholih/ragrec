"""Customer persona models and data structures."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Persona:
    """Customer persona with behavioral characteristics."""

    id: str
    name: str
    description: str
    size: int  # Number of customers
    avg_age: float | None = None
    top_categories: list[str] | None = None
    avg_purchases_per_customer: float | None = None
    avg_spend_per_customer: float | None = None
    club_member_ratio: float | None = None
    fashion_news_active_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert persona to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "size": self.size,
            "avg_age": self.avg_age,
            "top_categories": self.top_categories or [],
            "avg_purchases_per_customer": self.avg_purchases_per_customer,
            "avg_spend_per_customer": self.avg_spend_per_customer,
            "club_member_ratio": self.club_member_ratio,
            "fashion_news_active_ratio": self.fashion_news_active_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        """Create persona from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            size=data["size"],
            avg_age=data.get("avg_age"),
            top_categories=data.get("top_categories"),
            avg_purchases_per_customer=data.get("avg_purchases_per_customer"),
            avg_spend_per_customer=data.get("avg_spend_per_customer"),
            club_member_ratio=data.get("club_member_ratio"),
            fashion_news_active_ratio=data.get("fashion_news_active_ratio"),
        )


@dataclass
class CustomerPersonaAssignment:
    """Assignment of a customer to a persona."""

    customer_id: str
    persona_id: str
    embedding_cluster: int  # HDBSCAN cluster
    graph_community: int  # Louvain community
    confidence: float  # Assignment confidence (0-1)
