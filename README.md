# GenAI Neo4j Ollama - Thai Medicinal Terminology (TMT) Hybrid Retriever

A hybrid retrieval system for Thai Medicinal Terminology using Neo4j graph database and Ollama LLMs.

## 🏗️ Architecture

- **Neo4j**: Graph database for storing TMT hierarchical data (SUBS → VTM → GP → GPU → TP → TPU)
- **Ollama**: Local LLM and embedding service
- **Hybrid Search**: Combines vector similarity search with full-text search using Reciprocal Rank Fusion (RRF)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- (Optional) NVIDIA GPU for faster inference

### 1. Start Services

```bash
# Start Neo4j and Ollama
docker-compose up -d

# Check service status
docker-compose ps
```

### 2. Pull Ollama Models

```bash
# Pull the embedding model
docker exec -it ollama-service ollama pull bge-m3:latest

# Pull the LLM model
docker exec -it ollama-service ollama pull qwen2.5:7b-instruct
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update credentials if needed:

```bash
cp .env.example .env
```

### 4. Run the Application

```bash
python genai_neo4j_ollama.py
```

## 📊 Services

| Service | Port | URL |
|---------|------|-----|
| Neo4j Browser | 7474 | http://localhost:7474 |
| Neo4j Bolt | 7687 | bolt://localhost:7687 |
| Ollama API | 11434 | http://localhost:11434 |

## 🔧 Configuration

### Neo4j Settings

- Default user: `neo4j`
- Default password: Set in `.env` file
- Memory settings optimized for moderate workloads

### Ollama Models

- **Embedding**: `bge-m3:latest` (1024 dimensions)
- **LLM**: `qwen2.5:7b-instruct`

You can change models in the `.env` file.

## 📁 Data Volumes

- `neo4j_data`: Graph database storage
- `neo4j_logs`: Neo4j logs
- `neo4j_import`: Import files directory
- `ollama_data`: Model storage

## 🛠️ Maintenance

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f neo4j
docker-compose logs -f ollama
```

### Backup Neo4j Data

```bash
docker exec neo4j-tmt neo4j-admin database dump neo4j --to-path=/backup
```

### Reset Everything

```bash
docker-compose down -v  # WARNING: Deletes all data
```

## 🎯 Features

- **Hybrid Search**: Vector + Full-text search with RRF fusion
- **Thai Language Support**: Optimized for Thai pharmaceutical data
- **Graph Traversal**: Navigate TMT hierarchy relationships
- **Context-Aware LLM**: Grounded responses using graph context

## 🐛 Troubleshooting

### Neo4j won't start

Check memory limits and ensure port 7687 is not already in use.

### Ollama models not found

Make sure to pull the models after starting Ollama:

```bash
docker exec -it ollama-service ollama list
```

### GPU Support

Uncomment the GPU configuration in `docker-compose.yml` if you have NVIDIA GPU with Docker GPU support.

## 📝 License

(Add your license here)
