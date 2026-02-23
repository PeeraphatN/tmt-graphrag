from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

class TargetType(str, Enum):
    MANUFACTURER = "manufacturer"
    INGREDIENT = "ingredient"
    NLEM = "nlem"
    HIERARCHY = "hierarchy"
    FORMULA = "formula"
    GENERAL = "general"

class SearchStrategy(str, Enum):
    RETRIEVE = "retrieve"
    COUNT = "count"
    LIST = "list"
    VERIFY = "verify"


class RetrievalMode(str, Enum):
    VECTOR_HEAVY = "vector_heavy"
    BALANCED = "balanced"
    FULLTEXT_HEAVY = "fulltext_heavy"

class GraphRAGQuery(BaseModel):
    """
    Represents a structured search query for the GraphRAG system.
    Extracts the user's intent into explicit filters and search terms.
    """
    
    target_type: TargetType = Field(
        TargetType.GENERAL,
        description="The primary intent of the user question."
    )
    
    strategy: SearchStrategy = Field(
        SearchStrategy.RETRIEVE,
        description="The search strategy to use (retrieve, count, list, verify)."
    )

    query: str = Field(
        description="The extracted search term (entity name) to look up in the graph."
    )
    
    # Metadata Filters
    nlem_filter: Optional[bool] = Field(
        None, 
        description="Set to True if the user specifically asks for NLEM/National List of Essential Medicines/บัญชียาหลัก."
    )
    
    nlem_category: Optional[str] = Field(
        None,
        description="Specific NLEM category if mentioned (e.g. 'ง', 'ก', 'ข')."
    )

    manufacturer_filter: Optional[str] = Field(
        None,
        description="Specific manufacturer name to filter by, if mentioned."
    )
    
    limit: int = Field(
        10,
        description="Number of results to return."
    )

    # Adaptive retrieval profile (set during AQT)
    token_count: int = Field(
        0,
        description="Total token count for the user question."
    )

    entity_token_count: int = Field(
        0,
        description="Estimated count of entity-like tokens in the question."
    )

    entity_ratio: float = Field(
        0.0,
        description="entity_token_count / token_count. Used to adapt retrieval weights."
    )

    retrieval_mode: RetrievalMode = Field(
        RetrievalMode.BALANCED,
        description="Adaptive mode controlling vector/fulltext priority."
    )

    vector_weight: float = Field(
        0.5,
        description="Weight for vector ranking signal in weighted RRF."
    )

    fulltext_weight: float = Field(
        0.5,
        description="Weight for fulltext ranking signal in weighted RRF."
    )

    def to_cypher_filter(self) -> str:
        """Generates a Cypher WHERE clause fragment."""
        clauses = []
        if self.nlem_filter:
            clauses.append("n.nlem = true")
        if self.nlem_category:
            clauses.append(f"n.nlem_category = '{self.nlem_category}'")
        if self.manufacturer_filter:
            # Note: This might need fuzzy match in real implementation
            clauses.append(f"n.manufacturer CONTAINS '{self.manufacturer_filter}'")
            
        return " AND ".join(clauses) if clauses else "1=1"
