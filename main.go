package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

type QueryRequest struct {
	Query string `json:"query"`
}

func main() {
	// 1. Load Environment Variables
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, relying on system environment variables")
	}

	ctx := context.Background()

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY not found in .env file")
	}

	
	// 2. Initialize the LLM (Google Gemini)
	llm, err := googleai.New(ctx, 
		googleai.WithAPIKey(apiKey), 
		googleai.WithDefaultModel("gemini-2.5-flash"), // <-- CHANGE THIS TO 2.5
		googleai.WithDefaultEmbeddingModel("gemini-embedding-001"), 
	)
	if err != nil {
		log.Fatalf("Failed to create Gemini LLM: %v", err)
	}

	// 3. Initialize the Gemini Embedder
	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	// 4. Set up the Vector Store (ChromaDB) with Gemini Embeddings
	store, err := chroma.New(
		chroma.WithChromaURL("http://localhost:8000"),
		chroma.WithDistanceFunction("cosine"),
		chroma.WithNameSpace("corporate_policies"),
		chroma.WithEmbedder(embedder), // This forces Chroma to use Gemini instead of OpenAI
	)
	if err != nil {
		log.Fatalf("Failed to connect to Chroma: %v", err)
	}

	// 5. Ingest Documents
	fmt.Println("Ingesting security policies using Gemini embeddings...")
	file, err := os.Open("policies.txt")
	if err != nil {
		log.Fatalf("Failed to open policies file: %v", err)
	}
	defer file.Close()

	docLoaded := documentloaders.NewText(file)
	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = 500
	split.ChunkOverlap = 50

	docs, err := docLoaded.LoadAndSplit(ctx, split)
	if err != nil {
		log.Fatalf("Failed to split document: %v", err)
	}

	_, err = store.AddDocuments(ctx, docs)
	if err != nil {
		log.Fatalf("Failed to store documents in Chroma: %v", err)
	}
	fmt.Println("Policies ingested successfully!")

	// 6. Build the Retrieval QA Chain
	qaChain := chains.NewRetrievalQAFromLLM(
		llm,
		vectorstores.ToRetriever(store, 3), // Retrieve top 3 relevant chunks
	)

	// 7. Set up the API Router
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	r.Post("/api/v1/compliance/query", func(w http.ResponseWriter, r *http.Request) {
		var req QueryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
			return
		}

		// Run the query through our RAG chain
		answer, err := chains.Run(r.Context(), qaChain, req.Query)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to process query: %v", err), http.StatusInternalServerError)
			return
		}

		// Return the answer
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"answer": answer,
		})
	})

	// 8. Start the Server
	fmt.Println("Starting Security Policy API on :8080...")
	http.ListenAndServe(":8080", r)
}