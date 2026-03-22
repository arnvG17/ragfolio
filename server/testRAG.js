import { loadVectorStore } from "./utils/vectorStoreLoader.js";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv();

async function testArnavAI() {
  try {
    console.log("🧪 Testing Arnav AI without chat model...");
    
    const query = "What college does Arnav attend?";
    console.log(`❓ Question: "${query}"`);
    
    // Test embeddings directly
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "gemini-embedding-001",
      taskType: "RETRIEVAL_DOCUMENT",
      apiKey: process.env.GOOGLE_API_KEY,
    });
    
    const testEmbedding = await embeddings.embedDocuments(["test"]);
    console.log(`📏 Test embedding length: ${testEmbedding[0].length}`);
    
    const vectorStore = await loadVectorStore();
    const initialSearchResult = await vectorStore.maxMarginalRelevanceSearch(query, {
      k: 1,
      fetchK: 5,
    });
    
    console.log(`📊 Search results: ${initialSearchResult.length}`);
    if (initialSearchResult.length > 0) {
      console.log(`📏 First result score: ${initialSearchResult[0][1]}`);
      console.log(`📏 First result type: ${typeof initialSearchResult[0][1]}`);
    }
    
    const RELEVANCE_THRESHOLD = 0.69;
    
    if (initialSearchResult.length > 0 && initialSearchResult[0][1] >= RELEVANCE_THRESHOLD) {
      console.log(`✅ Arnav-related question detected. Score: ${initialSearchResult[0][1].toFixed(3)}`);
      
      const contextResults = await vectorStore.similaritySearch(query, 4);
      const context = contextResults
        .map((r, i) => `--- Document ${i + 1} ---\n${r.pageContent}`)
        .join("\n\n");
      
      console.log("📄 Context found:");
      console.log(context.substring(0, 500) + "...");
      
      // Simple answer extraction based on context
      const answer = extractAnswer(context, query);
      console.log(`✅ Answer: "${answer}"`);
    } else {
      const score = initialSearchResult.length > 0 ? initialSearchResult[0][1].toFixed(3) : "N/A";
      console.log(`❌ Unrelated question detected. Top Score: ${score}`);
    }
    
  } catch (error) {
    console.error("❌ Error:", error.message);
    console.error("❌ Full error:", error);
  }
}

function extractAnswer(context, query) {
  // Simple keyword-based answer extraction
  if (query.toLowerCase().includes("college")) {
    const lines = context.split('\n');
    for (const line of lines) {
      if (line.toLowerCase().includes("college") || line.toLowerCase().includes("engineering")) {
        return line.trim();
      }
    }
  }
  return "Based on the context, I found information about Arnav's education.";
}

testArnavAI();
