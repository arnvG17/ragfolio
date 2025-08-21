import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";

const router = express.Router();

const model = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_MODEL || "gemini-1.5-flash",
  apiKey: "AIzaSyBV_mBomUshBRX4FyuDBgOUiWTjON205Os",
});

router.post("/", async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    // Load vector store + run similarity search
    const vectorStore = await loadVectorStore();
    const results = await vectorStore.similaritySearch(query, 3);

    // Join context
    const context = results
      .map(
        (r, i) =>
          `Doc ${i + 1} (score ${r.score.toFixed(3)}):\n${r.pageContent}`
      )
      .join("\n\n");

    // Call Gemini chat model
    const response = await model.invoke([
      {
        role: "system",
        content:
          "You are a helpful assistant. Use the provided context if relevant.",
      },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${query}` },
    ]);

    res.json({ answer: response.content, sources: results });
  } catch (error) {
    console.error("💥 Chat error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

export default router;
