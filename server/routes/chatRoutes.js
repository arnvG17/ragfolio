import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";
import { config as configDotenv } from "dotenv";

configDotenv();

const router = express.Router();

const model = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_MODEL || "gemini-1.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
});

router.post("/", async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    // --- Step 1: Check for relevance using the vector store ---
    const vectorStore = await loadVectorStore();
    
    // First, search for the single best matching document to check its score.
    // similaritySearchWithScore returns an array of [Document, score] tuples.
    const initialSearchResult = await vectorStore.similaritySearchWithScore(query, 1);

    // --- Step 2: Decide which mode to use based on the score ---

    // This is the threshold for deciding if a question is "about" Lewis Hamilton.
    // You will need to tune this value based on your testing. Start with 0.75.
    // Higher (e.g., 0.8) is stricter, lower (e.g., 0.7) is more lenient.
    const RELEVANCE_THRESHOLD = 0.75; 

    let prompt;
    let isHamiltonRelated = false;

    // Check if we found any document AND if its score is above our threshold.
    if (initialSearchResult.length > 0 && initialSearchResult[0][1] >= RELEVANCE_THRESHOLD) {
      // --- MODE 1: The question IS related to Lewis Hamilton ---
      isHamiltonRelated = true;
      console.log(`✅ Hamilton-related question detected. Score: ${initialSearchResult[0][1].toFixed(3)}`);

      // Now that we know it's relevant, get a few more documents for richer context.
      const contextResults = await vectorStore.similaritySearch(query, 4);
      const context = contextResults
        .map((r, i) => `--- Document ${i + 1} ---\n${r.pageContent}`)
        .join("\n\n");
      
      prompt = `
        # Your Persona: Lewis Hamilton Expert
        You are Hamilton AI, a friendly and conversational expert on Lewis Hamilton.
        Your answers about Lewis Hamilton MUST be based exclusively on the information within the provided 'Context'.
        You must act as if you know this information yourself. Never mention the context or documents.
        
        # Universal Rule
        NO SOURCES OR CITATIONS. Just provide a clean, direct answer.

        # Context
        ${context}

        # Question
        ${query}
      `;

    } else {
      // --- MODE 2: The question is NOT related to Lewis Hamilton ---
      isHamiltonRelated = false;
      const score = initialSearchResult.length > 0 ? initialSearchResult[0][1].toFixed(3) : "N/A";
      console.log(`❌ Unrelated question detected. Top Score: ${score}`);

      prompt = `
        # Your Persona: General Knowledge Assistant
        You are Hamilton AI, a friendly and helpful AI assistant.
        The user has asked a question that is not about your primary subject, Lewis Hamilton.
        Provide a full, accurate, and helpful answer to the user's question using your general knowledge.

        # Universal Rule
        NO SOURCES OR CITATIONS. Just provide a clean, direct answer.

        # Question
        ${query}
      `;
    }

    // --- Step 3: Invoke the model with the dynamically created prompt ---
    const response = await model.invoke(prompt);

    res.json({ 
      answer: response.content,
      isHamiltonRelated: isHamiltonRelated // Optional: send this to your frontend for debugging
    });

  } catch (error) {
    console.error("💥 Chat error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

export default router;