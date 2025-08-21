import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv(); // ✅ loads .env into process.env

// Initialize Gemini embeddings
export const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: "embedding-001", // optimized for vector search
  apiKey: process.env.GOOGLE_API_KEY, // ✅ now pulled from .env
});
