from typing import Optional, List, Literal
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