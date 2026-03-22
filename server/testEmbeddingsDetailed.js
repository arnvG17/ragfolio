import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv();

async function testEmbeddings() {
  try {
    console.log("🧪 Testing embeddings in detail...");
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "gemini-embedding-001",
      taskType: "RETRIEVAL_DOCUMENT",
      apiKey: process.env.GOOGLE_API_KEY,
    });
    
    console.log("✅ Embeddings initialized");
    
    // Test with different methods
    console.log("🧪 Testing embedQuery...");
    const queryResult = await embeddings.embedQuery("test query");
    console.log(`📏 Query result length: ${queryResult.length}`);
    console.log(`📏 First few values: ${queryResult.slice(0, 5)}`);
    
    console.log("🧪 Testing embedDocuments...");
    const docResult = await embeddings.embedDocuments(["test document"]);
    console.log(`📏 Doc result length: ${docResult[0].length}`);
    console.log(`📏 First few values: ${docResult[0].slice(0, 5)}`);
    
  } catch (error) {
    console.error("❌ Error:", error.message);
    console.error("❌ Full error:", error);
  }
}

testEmbeddings();
