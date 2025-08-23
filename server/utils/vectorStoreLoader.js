import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv(); // ✅ Load .env variables

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Cosine similarity function to score documents
function cosineSimilarity(a, b) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Main function to load and expose our custom vector store
export async function loadVectorStore() {
  try {
    const filePath = path.join(__dirname, "../vectorStore.json");
    const raw = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(raw);

    if (!Array.isArray(data)) {
      throw new Error("vectorStore.json must be an array of documents");
    }

    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "embedding-001",
      apiKey: process.env.GOOGLE_API_KEY,
    });

    // Internal helper function to embed and score all documents
    const _search = async (query) => {
      const [queryEmbedding] = await embeddings.embedDocuments([query]);
      return data.map((doc) => ({
        ...doc,
        score: cosineSimilarity(queryEmbedding, doc.embedding),
      }));
    };

    // The object returned by loadVectorStore
    return {
      // Standard similaritySearch method
      similaritySearch: async (query, k = 3) => {
        const scored = await _search(query);
        return scored
          .sort((a, b) => b.score - a.score)
          .slice(0, k)
          .map((doc) => ({
            pageContent: doc.pageContent,
            metadata: doc.metadata,
            score: doc.score,
          }));
      },

      // THIS IS THE METHOD THAT WAS MISSING
      // It returns results in the [Document, score] format
      // that chatRoutes.js now expects.
      maxMarginalRelevanceSearch: async (query, { k = 1, fetchK = 5 }) => {
        const scored = await _search(query);
        const sorted = scored.sort((a, b) => b.score - a.score);
        
        // We only need the top-k results for the relevance check
        const topK = sorted.slice(0, k);

        // Return the results in the [Document, score] format
        return topK.map((doc) => [{
          pageContent: doc.pageContent,
          metadata: doc.metadata
        }, doc.score]);
      },

      // It's also a good idea to add this method for future compatibility
      similaritySearchWithScore: async (query, k = 3) => {
        const scored = await _search(query);
        const sorted = scored.sort((a, b) => b.score - a.score);
        
        return sorted.slice(0, k).map((doc) => [{
          pageContent: doc.pageContent,
          metadata: doc.metadata
        }, doc.score]);
      }
    };

  } catch (err) {
    console.error("❌ Error reading vectorStore.json:", err.message);
    throw new Error("Vector store unavailable");
  }
}