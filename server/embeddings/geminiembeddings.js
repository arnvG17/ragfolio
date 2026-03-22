import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv(); // ✅ loads .env into process.env

// Initialize Gemini embeddings
export const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  modelName: "gemini-embedding-001", // Current 2026 stable standard
  taskType: "RETRIEVAL_DOCUMENT", // Optimize for document retrieval
});
