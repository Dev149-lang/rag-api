# RAG-Powered Security Policy Compliance API

An enterprise-grade, Retrieval-Augmented Generation (RAG) API built in Go. It ingests corporate security policies into a local ChromaDB vector database and uses Google's Gemini LLM to enforce compliance, answer queries, and provide exact source citations.

## Prerequisites
* [Go 1.21+](https://go.dev/dl/)
* [Docker](https://www.docker.com/products/docker-desktop/) (for running the vector database)
* A [Google Gemini API Key](https://aistudio.google.com/)



## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/rag-api.git](https://github.com/YOUR_USERNAME/rag-api.git)
cd rag-api


**2. Copy the env file and add your gemini key**

Bash
cp .env.example .env

**3. Start the vector db
Spin up a local instance of ChromaDB using Docker:

docker run -d -p 8000:8000 chromadb/chroma:0.5.20

Download the Go dependencies and start the server. It will automatically ingest the policies.txt file on startup:

go mod tidy
go run main.go



You can test the endpoint using PowerShell or cURL.

PowerShell:

PowerShell
Invoke-RestMethod -Uri http://localhost:8080/api/v1/compliance/query `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"query": "How often do I need to rotate my password?"}'