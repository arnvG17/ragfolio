import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { configDotenv } from "dotenv";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


// Cosine similarity function
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

export async function loadVectorStore() {
  try {
    const filePath = path.join(__dirname, "../vectorStore.json");
    const raw = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(raw); // array of { pageContent, metadata, embedding }

    if (!Array.isArray(data)) {
      throw new Error("vectorStore.json must be an array of documents");
    }

    // Init Gemini embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "embedding-001",
      apiKey: "AIzaSyBV_mBomUshBRX4FyuDBgOUiWTjON205Os",
    });

    return {
      similaritySearch: async (query, k = 3) => {
        // 1️⃣ Embed the user query using Gemini
        const [queryEmbedding] = await embeddings.embedDocuments([query]);

        // 2️⃣ Score each document
        const scored = data.map((doc) => ({
          ...doc,
          score: cosineSimilarity(queryEmbedding, doc.embedding),
        }));

        // 3️⃣ Sort and pick top-k
        return scored
          .sort((a, b) => b.score - a.score)
          .slice(0, k)
          .map((doc) => ({
            pageContent: doc.pageContent,
            metadata: doc.metadata,
            score: doc.score,
          }));
      },
    };
  } catch (err) {
    console.error("❌ Error reading vectorStore.json:", err.message);
    throw new Error("Vector store unavailable");
  }
}
