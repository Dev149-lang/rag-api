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
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

type QueryRequest struct {
	Query string `json:"query"`
}

//j
type QueryResponse struct {
	Answer  string   `json:"answer"`
	Sources []string `json:"sources"`
}

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, relying on system environment variables")
	}

	ctx := context.Background()

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY not found in .env file")
	}

	llm, err := googleai.New(ctx, 
		googleai.WithAPIKey(apiKey), 
		googleai.WithDefaultModel("gemini-2.5-flash"), 
		googleai.WithDefaultEmbeddingModel("gemini-embedding-001"), 
	)
	if err != nil {
		log.Fatalf("Failed to create Gemini LLM: %v", err)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	store, err := chroma.New(
		chroma.WithChromaURL("http://localhost:8000"),
		chroma.WithDistanceFunction("cosine"),
		chroma.WithNameSpace("corporate_policies"),
		chroma.WithEmbedder(embedder), 
	)
	if err != nil {
		log.Fatalf("Failed to connect to Chroma: %v", err)
	}

	fmt.Println("Ingesting security policies...")
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

	qaChain := chains.NewRetrievalQAFromLLM(
		llm,
		vectorstores.ToRetriever(store, 3), 
	)
	// 2. Tell the chain to preserve the documents it retrieved
	qaChain.ReturnSourceDocuments = true 

	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	r.Post("/api/v1/compliance/query", func(w http.ResponseWriter, r *http.Request) {
		var req QueryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
			return
		}

		// 3. Use chains.Call instead of chains.Run
		res, err := chains.Call(r.Context(), qaChain, map[string]any{
			"query": req.Query,
		})
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to process query: %v", err), http.StatusInternalServerError)
			return
		}

		// extraction block
		var answer string
		if val, ok := res["text"].(string); ok {
			answer = val
		} else if val, ok := res["result"].(string); ok {
			answer = val
		} else {
			http.Error(w, "Failed to parse the AI response format", http.StatusInternalServerError)
			return
		}
		
		// 4. Extract the source documents and format them for the API response
		var sourceList []string
		if docs, ok := res["source_documents"].([]schema.Document); ok {
			for i, doc := range docs {
				// Grab the filename from the metadata (if it exists)
				fileName := "Unknown Source"
				if src, exists := doc.Metadata["source"]; exists {
					fileName = fmt.Sprintf("%v", src)
				}
				
				// Grab a quick snippet of the text that matched
				snippet := doc.PageContent
				if len(snippet) > 60 {
					snippet = snippet[:60] + "..."
				}
				
				sourceList = append(sourceList, fmt.Sprintf("%s (Match %d) - \"%s\"", fileName, i+1, snippet))
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(QueryResponse{
			Answer:  answer,
			Sources: sourceList,
		})
	})

	fmt.Println("Starting Security Policy API on :8080...")
	http.ListenAndServe(":8080", r)
}