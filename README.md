# TMT GraphRAG: Thai Medicinal Terminology & Intelligent Search

**TMT GraphRAG** is an advanced **Retrieval-Augmented Generation (RAG)** system designed for Thai pharmaceutical data. It leverages a **Neo4j Knowledge Graph** to understand complex drug relationships (TMT Hierarchy) and provide precise answers about drug properties, reimbursement status (NLEM), and manufacturers.

## 🌟 Key Features

### 1. Hybrid Search (Vector + Graph + Fulltext)
Combines three search methodologies for maximum recall and precision:
-   **Vector Search**: Semantic similarity using `bge-m3` embeddings.
-   **Fulltext Search**: Exact keyword matching for Thai drug names.
-   **Graph Traversal**: Expands context by traversing relationships (e.g., finding all brands of a generic drug).

### 2. Intelligent Intent Router (AQT)
The system doesn't just "search" blindly. It classifies User Intent into strategies:
-   **RETRIEVE**: General information lookup (e.g., "Properties of Paracetamol").
-   **COUNT**: Quantitative analysis (e.g., "How many drugs in NLEM Category A?").
-   **LIST**: Listing items based on criteria (e.g., "List all GPO products").
-   **VERIFY**: Fact-checking validity (e.g., "Is Paracap reimbursable?").

### 3. Deep Filtering
Successfully handles complex constraints:
-   **NLEM Status**: Checks if a drug is in the *National List of Essential Medicines*.
-   **Manufacturer**: Filters results by specific companies (e.g., "GPO", "Siam Bheasach").

---

## 🏗️ System Architecture

The pipeline consists of modular services located in `src/`:

```mermaid
graph LR
    User[User Question] --> API[FastAPI /chat]
    API --> Pipeline[GraphRAGPipeline]
    
    subgraph Core Pipeline
        Pipeline --> AQT[AQT Service / Intent Router]
        AQT --> Search[Search Service]
        Search --> Extract[Extraction Service]
        Extract --> Format[Formatting Service]
    end
    
    Search --> Neo4j[(Neo4j Graph DB)]
    Format --> LLM[Ollama LLM]
    
    AQT -.->|Filters (NLEM/Mfr)| Search
```

### Module Breakdown (`src/`)

-   **`pipeline.py`**: Orchestrates the data flow (Transform -> Search -> Extract -> Format).
-   **`services/aqt.py`**: **(Brain)** Uses LLM to extract `TargetType` (Intent) and `Filters` from natural language.
-   **`services/search.py`**: **(Retriever)** Executes `hybrid_search` and `expand_context` (Graph Traversal). Applies filters dynamically.
-   **`services/extraction.py`**: **(Parser)** Deterministically cleans and structures the graph data into JSON entities for the LLM.
-   **`services/formatting.py`**: **(Generator)** Generates the final Thai response based *only* on the extracted JSON context.
-   **`prompts/templates.py`**: Centralized Prompt Engineering (System Prompts for Classification & Formatting).

---

## 🚀 Getting Started

### Prerequisites
-   **Docker & Docker Compose**
-   **Python 3.10+**
-   **Ollama** (running locally or in container)

### 1. Setup Services
Start Neo4j and Ollama containers:
```bash
docker-compose up -d
```

### 2. Configuration (`.env`)
Ensure your `.env` file has the correct Neo4j and Ollama settings:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b-instruct
EMBED_MODEL=bge-m3:latest
```

### 3. Run the API
Start the FastAPI server:
```bash
python src/api/main.py
```
*API will run at `http://localhost:8000`*

### 4. Interactive Testing
Use the test script to verify the Intent Router:
```bash
python scripts/test_intent_router.py
```

---

## 📂 Project Structure

```text
.
├── src/
│   ├── api/                # FastAPI Endpoints
│   ├── services/           # Core Business Logic (AQT, Search, Extract)
│   ├── schemas/            # Pydantic Models (Query, Response)
│   ├── models/             # LLM & Embedding Wrappers
│   ├── prompts/            # LLM Prompt Templates
│   ├── cache/              # Semantic Caching Layer
│   ├── config.py           # Environment Config
│   └── pipeline.py         # Main Pipeline Class
├── scripts/                # Utility & Verification Scripts
├── tests/                  # Unit Tests
├── docker-compose.yml      # Service Orchestration
└── README.md               # Project Documentation
```

## 🧠 Knowledge Graph Schema (Simplified)

The system is built on **TMT (Thai Medicinal Terminology)** structure:
-   **SUB / VTM**: Generic Substances
-   **GP / GPU**: Generic Products (Dosage Forms)
-   **TP / TPU**: Trade Products (Brand Names)
-   **Properties**: `fsn` (FullName), `manufacturer`, `nlem` (True/False), `active_ingredient`
