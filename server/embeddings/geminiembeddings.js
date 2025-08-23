import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";
configDotenv();

export const embeddings = new GoogleGenerativeAIEmbeddings({
  // recommended model for text embeddings
  modelName: "gemini-embedding-001", // ← use this
  apiKey: process.env.GOOGLE_API_KEY,
  // optional LangChain params to tune throughput / reliability
  batchSize: 32,         // tune 8..100 depending on your environment
  // output_dimensionality: 1024 // optional: reduce dims if supported by your SDK
});
