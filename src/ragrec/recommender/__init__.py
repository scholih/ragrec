"""Recommendation engines (visual, collaborative, fusion)."""

from ragrec.recommender.collaborative import CollaborativeRecommender
from ragrec.recommender.visual import VisualRecommender

__all__ = ["VisualRecommender", "CollaborativeRecommender"]
