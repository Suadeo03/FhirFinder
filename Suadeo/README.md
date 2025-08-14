# FHIR Registry - V2-FHIR Mapping Search System

A comprehensive search platform for HL7 FHIR mappings with AI-powered insights and semantic search capabilities.

## 🌟 Features

- **Semantic Search**: Vector-based similarity search using ChromaDB and sentence transformers
- **V2-FHIR Mappings**: Upload, process, and search HL7 V2 to FHIR mapping datasets
- **AI Summaries**: LLM-powered search result summaries via Ollama integration
- **Query Analytics**: Comprehensive tracking of search performance and user interactions
- **Feedback Learning**: Machine learning system that improves search results based on user feedback
- **Real-time Caching**: Redis-powered caching for improved performance
- **Health Monitoring**: Built-in health checks for all services
- **Web Interface**: User-friendly frontend for searching and managing mappings

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   PostgreSQL    │
│   (Static)      │◄──►│   Backend       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ChromaDB      │    │     Redis       │    │     Ollama      │
│ (Vector Search) │    │   (Caching)     │    │   (LLM/AI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fhir-registry
```

### 2. Start the Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f fhir_app
```

### 3. Setup Ollama Model

```bash
# Pull the TinyLlama model
chmod +x setup_ollama.sh
./setup_ollama.sh docker
```

### 4. Access the Application

- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Database Admin**: http://localhost:8080 (PgAdmin)

## 📁 Project Structure

```
fhir-registry/
├── docker-compose.yml          # Container orchestration
├── setup_ollama.sh            # LLM model setup script
├── README.md                  # This file
├── backend/
│   ├── Dockerfile            # Backend container configuration
│   ├── requirements.txt      # Python dependencies
│   ├── main.py              # FastAPI application entry point
│   ├── static/              # Frontend files (HTML, JS, CSS)
│   │   ├── index.html
│   │   └── script.js
│   ├── api/                 # API route definitions
│   │   └── v1/
│   │       └── endpoints/
│   ├── models/              # Database models
│   │   └── database/
│   ├── services/            # Business logic
│   │   ├── search/          # Search services
│   │   ├── analytics/       # Query tracking
│   │   └── etl/            # Data processing
│   └── config/             # Configuration files
│       ├── database.py
│       ├── chroma.py
│       └── redis_cache.py
└── init-scripts/           # Database initialization
    └── init.sql
```

## 🔧 Configuration

### Environment Variables

The application uses environment variables for configuration:

```bash
# Database
DATABASE_URL=postgresql://fhir_user:admin@postgres:5432/fhir_registry
POSTGRES_HOST=postgres
POSTGRES_DB=fhir_registry
POSTGRES_USER=fhir_user
POSTGRES_PASSWORD=admin

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379

# ChromaDB Vector Database
CHROMA_HOST=chroma
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=fhir_profiles

# Ollama LLM
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=tinyllama

# Application
ENVIRONMENT=production
DEBUG=false
```

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| FastAPI App | 8000 | Main application and API |
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Caching layer |
| ChromaDB | 8001 | Vector database |
| Ollama | 11434 | LLM service |
| PgAdmin | 8080 | Database administration |

## 📊 API Endpoints

### V2-FHIR Dataset Management

```bash
# Upload dataset
POST /api/v1/v2-fhir-datasets/upload

# Process dataset  
POST /api/v1/v2-fhir-datasets/{id}/process

# Activate dataset
PUT /api/v1/v2-fhir-datasets/{id}/activate

# List datasets
GET /api/v1/v2-fhir-datasets
```

### Search & Analytics

```bash
# Search V2-FHIR mappings
POST /api/v1/search/v2-fhir-mappings

# Record feedback
POST /api/v1/search/v2-fhir-mappings/feedback

# Get analytics
GET /api/v1/search/v2-fhir-mappings/analytics

# Search suggestions
GET /api/v1/search/v2-fhir-mappings/suggest
```

### System Health

```bash
# Basic health check
GET /api/v1/health

# Detailed health information
GET /api/v1/health/detailed
```

## 🔍 Usage Guide

### 1. Upload V2-FHIR Mappings

1. Navigate to the web interface at http://localhost:8000
2. Use the upload form to submit Excel/CSV files containing V2-FHIR mappings
3. Expected columns:
   - `id`, `local_id`, `resource`, `sub_detail`
   - `fhir_detail`, `fhir_version`
   - `hl7v2_field`, `hl7v2_field_detail`, `hl7v2_field_version`

### 2. Process and Activate Datasets

```bash
# Process uploaded dataset
curl -X POST "http://localhost:8000/api/v1/v2-fhir-datasets/{dataset_id}/process"

# Activate for searching
curl -X PUT "http://localhost:8000/api/v1/v2-fhir-datasets/{dataset_id}/activate"
```

### 3. Search Mappings

Use the web interface or API:

```bash
curl -X POST "http://localhost:8000/api/v1/search/v2-fhir-mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "patient identifier",
    "limit": 5,
    "include_summary": true,
    "only_active": true
  }'
```

### 4. Provide Feedback

Improve search results by providing feedback:

```bash
curl -X POST "http://localhost:8000/api/v1/search/v2-fhir-mappings/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "patient name",
    "mapping_id": "mapping-123",
    "feedback_type": "positive"
  }'
```

## 📈 Analytics & Monitoring

### Query Analytics

The system automatically tracks:
- Search performance metrics
- Result relevance scores
- User interaction patterns
- Most common queries
- Success/failure rates

### Health Monitoring

Monitor system health via:
- `/health` - Quick health status
- `/health/detailed` - Comprehensive diagnostics
- Docker health checks
- Service connectivity status

### Performance Optimization

The system includes:
- **Redis caching** for frequently searched queries
- **Vector indexing** for fast semantic search
- **Query optimization** based on user feedback
- **Background processing** for large datasets

## 🛠️ Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Run locally
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Database Management

```bash
# Connect to PostgreSQL
docker exec -it fhir_postgres psql -U fhir_user -d fhir_registry

# Access PgAdmin
# URL: http://localhost:8080
# Email: admin@example.com
# Password: admin_password
```

### Adding New Features

1. **API Endpoints**: Add routes in `backend/api/v1/endpoints/`
2. **Database Models**: Define in `backend/models/database/`
3. **Business Logic**: Implement in `backend/services/`
4. **Frontend**: Update files in `backend/static/`

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service_name]

# Restart specific service
docker-compose restart fhir_app

# Rebuild after code changes
docker-compose up --build -d

# Stop all services
docker-compose down

# Clean up volumes (⚠️ deletes data)
docker-compose down -v
```

## 🔒 Security Considerations

- Change default passwords in production
- Use environment variables for sensitive configuration
- Enable HTTPS for production deployments
- Implement proper authentication and authorization
- Regular security updates for dependencies

## 🚨 Troubleshooting

### Common Issues

1. **Port 8000 already in use**:
   ```bash
   sudo lsof -i :8000
   kill -9 PID
   ```

2. **Database connection issues**:
   ```bash
   docker-compose logs postgres
   docker-compose restart postgres
   ```

3. **ChromaDB not responding**:
   ```bash
   docker-compose logs chroma
   curl http://localhost:8001/api/v1/heartbeat
   ```

4. **Ollama model not loaded**:
   ```bash
   docker exec fhir_ollama ollama list
   ./setup_ollama.sh docker
   ```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment variable
export DEBUG=true

# Or in docker-compose.yml
environment:
  - DEBUG=true
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

See License

## 🤝 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

**Built with FastAPI, PostgreSQL, ChromaDB, Redis, and Ollama for intelligent V2-FHIR mapping search.**
