import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

try {
  console.log("🔄 Loading vector store...");
  const filePath = path.join(__dirname, "../vectorStore.json");
  console.log(`📁 Reading file: ${filePath}`);
  const raw = await fs.readFile(filePath, "utf-8");
  const data = JSON.parse(raw);
  console.log(`📊 Loaded ${data.length} documents`);

  if (!Array.isArray(data)) {
    throw new Error("vectorStore.json must be an array of documents");
  }

  console.log("🔧 Initializing embeddings...");
  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "gemini-embedding-001",
    taskType: "RETRIEVAL_DOCUMENT",
    apiKey: process.env.GOOGLE_API_KEY,
  });
  console.log("✅ Embeddings initialized");

  console.log("🧪 Testing search...");
  const [queryEmbedding] = await embeddings.embedDocuments(["What college does Arnav attend?"]);
  console.log("📏 Query embedding length:", queryEmbedding.length);

  if (data.length > 0) {
    console.log("📄 First doc embedding length:", data[0].embedding.length);
    console.log("📄 First doc content preview:", data[0].pageContent.substring(0, 100) + "...");
  }

} catch (error) {
  console.error("❌ Error:", error.message);
  console.error("❌ Full error:", error);
}
