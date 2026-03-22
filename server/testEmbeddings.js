import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv();

console.log("🔑 Testing API key:", process.env.GOOGLE_API_KEY ? "Found" : "Not found");

try {
  console.log("🔧 Initializing embeddings...");
  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "gemini-embedding-001",
    taskType: "RETRIEVAL_DOCUMENT",
    apiKey: process.env.GOOGLE_API_KEY,
  });
  console.log("✅ Embeddings initialized");

  console.log("🧪 Testing embedding...");
  const result = await embeddings.embedDocuments(["test"]);
  console.log("✅ Embedding successful, length:", result[0].length);
} catch (error) {
  console.error("❌ Error:", error.message);
  console.error("❌ Full error:", error);
}
