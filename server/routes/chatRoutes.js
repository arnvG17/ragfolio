import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";
import { config as configDotenv } from "dotenv";

configDotenv();

const router = express.Router();

const model = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_MODEL || "gemini-1.5-flash",
  apiKey: process.env.GOOGLE_API_KEY2,
});

// --- Simple Intent Router for Greetings and Identity ---
const simpleRoutes = {
  "hi": "I am Hamilton AI, here to give you information on Lewis Hamilton. How can I help?",
  "hello": "I am Hamilton AI, here to give you information on Lewis Hamilton. How can I help?",
  "hey": "I am Hamilton AI, here to give you information on Lewis Hamilton. How can I help?",
  "sup": "I am Hamilton AI, here to give you information on Lewis Hamilton. How can I help?",
  "what up": "I am Hamilton AI, here to give you information on Lewis Hamilton. How can I help?",
  "who are you": "I am Hamilton AI, a chatbot designed to provide information about the life and career of Lewis Hamilton.",
  "what are you": "I am Hamilton AI, a chatbot designed to provide information about the life and career of Lewis Hamilton.",
};

// --- Pronoun Replacement Regex ---
const pronounRegex = /\b(he|him|his)\b/i; // Matches "he", "him", "his" case-insensitively

router.post("/", async (req, res) => {
  try {
    // The history from the client must now include the `isHamiltonRelated` flag
    // on each AI message to enable this logic.
    const { query, history = [] } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    const normalizedQuery = query.toLowerCase().trim();
    if (simpleRoutes[normalizedQuery]) {
      console.log("✅ Greeting/Identity route triggered.");
      return res.json({ answer: simpleRoutes[normalizedQuery] });
    }

    // --- Programmatic Pronoun Resolution ---
    let standaloneQuery = query;
    let lastAITurn = history.length > 0 ? history[history.length - 1] : null;

    if (lastAITurn && lastAITurn.type === 'ai' && lastAITurn.isHamiltonRelated) {
      if (pronounRegex.test(query)) {
        // If the last AI message was about Hamilton,
        // and the new query contains a pronoun, replace it.
        standaloneQuery = query.replace(pronounRegex, "Lewis Hamilton");
        console.log(`✨ Pronoun replaced. New query: "${standaloneQuery}"`);
      }
    }

    // --- Check Relevance using the new standalone query ---
    const vectorStore = await loadVectorStore();
    const initialSearchResult = await vectorStore.maxMarginalRelevanceSearch(standaloneQuery, {
      k: 1,
      fetchK: 5,
    });
    
    const RELEVANCE_THRESHOLD = 0.69; 
    let finalPrompt;
    let isHamiltonRelated = false;

    if (initialSearchResult.length > 0 && initialSearchResult[0][1] >= RELEVANCE_THRESHOLD) {
      isHamiltonRelated = true;
      console.log(`✅ Hamilton-related question detected. Score: ${initialSearchResult[0][1].toFixed(3)}`);

      const contextResults = await vectorStore.similaritySearch(standaloneQuery, 4);
      const context = contextResults
        .map((r, i) => `--- Document ${i + 1} ---\n${r.pageContent}`)
        .join("\n\n");
      
      finalPrompt = `
        # Your Persona: Lewis Hamilton Expert
        You are Hamilton AI. Your answers about Lewis Hamilton MUST be based exclusively on the 'Context'.
        Act as if you know this information yourself. Never mention the context or documents.
        NO SOURCES OR CITATIONS.
        
        # Context
        ${context}

        # Question
        ${standaloneQuery}
      `;
    } else {
      isHamiltonRelated = false;
      const score = initialSearchResult.length > 0 ? initialSearchResult[0][1].toFixed(3) : "N/A";
      console.log(`❌ Unrelated question detected. Top Score: ${score}`);

      finalPrompt = `
        # Your Persona: General Knowledge Assistant
        You are Hamilton AI. The user has asked a question that is not about your primary subject, Lewis Hamilton.
        Provide a full, accurate, and helpful answer to the user's question using your general knowledge.
        NO SOURCES OR CITATIONS.

        # Question
        ${standaloneQuery}
      `;
    }

    const finalResponse = await model.invoke(finalPrompt);

    res.json({ 
      answer: finalResponse.content,
      isHamiltonRelated: isHamiltonRelated
    });

  } catch (error) {
    console.error("💥 Chat error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

export default router;