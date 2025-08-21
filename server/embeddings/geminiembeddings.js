import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

// Initialize Gemini embeddings
export const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: "embedding-001", // optimized for vector search
  apiKey: "AIzaSyBV_mBomUshBRX4FyuDBgOUiWTjON205Os", // load from .env
});
