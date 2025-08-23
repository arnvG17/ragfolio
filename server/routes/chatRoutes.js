import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";
import { config as configDotenv } from "dotenv";

configDotenv();

const router = express.Router();

const model = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_MODEL || "gemini-1.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
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

router.post("/", async (req, res) => {
  try {
    const { query, history = [] } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    const normalizedQuery = query.toLowerCase().trim();
    if (simpleRoutes[normalizedQuery]) {
      console.log("✅ Greeting/Identity route triggered.");
      return res.json({ answer: simpleRoutes[normalizedQuery] });
    }

    let standaloneQuery = query;
    if (history.length > 0) {
      console.log("🤔 History detected. Condensing question...");
      const chatHistory = history.map(msg => 
        msg.type === 'human' ? new HumanMessage(msg.text) : new AIMessage(msg.text)
      );

      const condensingPrompt = `Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the chat history.

      Chat History:
      ${chatHistory.map(msg => `${msg._getType()}: ${msg.content}`).join('\n')}

      Follow-Up Question: ${query}

      Standalone question:`;

      const condensedResponse = await model.invoke(condensingPrompt);
      standaloneQuery = condensedResponse.content.trim(); // Added .trim() for safety
      console.log(`Condensing complete. Standalone Question: "${standaloneQuery}"`);
    }

    const vectorStore = await loadVectorStore();
    
    // ===================================================================
    //  THE FIX IS HERE: Replaced the failing function call
    // ===================================================================
    // OLD LINE THAT FAILED:
    // const initialSearchResult = await vectorStore.similaritySearchWithScore(standaloneQuery, 1);
    
    // NEW, MORE RELIABLE LINE:
    const initialSearchResult = await vectorStore.maxMarginalRelevanceSearch(standaloneQuery, {
      k: 1, // We only want the single best result
      fetchK: 5, // Fetch 5 candidates to select the best one from
    });
    // ===================================================================

    const RELEVANCE_THRESHOLD = 0.75; 
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
        You are Hamilton AI, a friendly and helpful AI assistant.
        The user has asked a question that is not about your primary subject, Lewis Hamilton.
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