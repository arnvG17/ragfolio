import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { config as configDotenv } from "dotenv";

configDotenv();

const router = express.Router();

const model = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_MODEL || "gemini-2.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
});

const simpleRoutes = {
  "hi": "I am Arnav AI, here to give you information on Arnav Gawandi. How can I help?",
  "hello": "I am Arnav AI, here to give you information on Arnav Gawandi. How can I help?",
  "hey": "I am Arnav AI, here to give you information on Arnav Gawandi. How can I help?",
  "sup": "I am Arnav AI, here to give you information on Arnav Gawandi. How can I help?",
  "what up": "I am Arnav AI, here to give you information on Arnav Gawandi. How can I help?",
  "who are you": "I am Arnav AI, a chatbot designed to provide information about Arnav Gawandi's life, education, experience, and interests.",
  "what are you": "I am Arnav AI, a chatbot designed to provide information about Arnav Gawandi's life, education, experience, and interests.",
};

// We don't need the simple pronoun regex anymore, the LLM will handle it.
// const pronounRegex = /\b(he|him|his)\b/i;

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

    // --- Start of NEW, More Robust Context Resolution ---
    // If there is a chat history, we always try to condense the query.
    if (history.length > 0) {
      console.log("🤔 History detected. Condensing query for context...");

      const chatHistory = history.map(msg => 
        msg.type === 'human' ? new HumanMessage(msg.text) : new AIMessage(msg.text)
      );

      // A single, robust prompt to handle all kinds of follow-ups.
      const condensingPrompt = [
        new HumanMessage(`Given the chat history and a follow-up question, rephrase the follow-up question into a single, standalone question that can be understood without the chat history. If the follow-up question is already standalone, return it as is.

        Chat History:
        human: What college does Arnav attend?
        ai: He attends Dwarkadas J. Sanghvi College of Engineering (DJSCE) in Mumbai.
        
        Follow-Up Question: What is he studying?
        
        Standalone question:`),
        new AIMessage("What is Arnav Gawandi studying at DJSCE?"),

        new HumanMessage(`Chat History:
        human: What technical clubs is Arnav part of?
        ai: He's part of DJS S4DS and DJS CODEAI clubs.

        Follow-Up Question: What does he do there?
        
        Standalone question:`),
        new AIMessage("What does Arnav Gawandi do in the DJS S4DS and DJS CODEAI clubs?"),

        new HumanMessage(`Chat History:
        ${chatHistory.map(msg => `${msg.type}: ${msg.text}`).join('\n')}

        Follow-Up Question: ${query}
        
        Standalone question:`),
      ];

      const condensedResponse = await model.invoke(condensingPrompt);
      standaloneQuery = condensedResponse.content.trim();
      
      // Log if the query was changed.
      if (standaloneQuery.toLowerCase() !== query.toLowerCase()) {
        console.log(`✨ Query condensed. Standalone Question: "${standaloneQuery}"`);
      } else {
        console.log("✅ Query was already standalone.");
      }
    }
    // --- End of NEW Context Resolution ---

    const vectorStore = await loadVectorStore();
    const initialSearchResult = await vectorStore.maxMarginalRelevanceSearch(standaloneQuery, {
      k: 1,
      fetchK: 5,
    });
    
    const RELEVANCE_THRESHOLD = 0.69; 
    let finalPrompt;
    let isArnavRelated = false;

    if (initialSearchResult.length > 0 && initialSearchResult[0][1] >= RELEVANCE_THRESHOLD) {
      isArnavRelated = true;
      console.log(`✅ Arnav-related question detected. Score: ${initialSearchResult[0][1].toFixed(3)}`);

      const contextResults = await vectorStore.similaritySearch(standaloneQuery, 4);
      const context = contextResults
        .map((r, i) => `--- Document ${i + 1} ---\n${r.pageContent}`)
        .join("\n\n");
      
      finalPrompt = `
        # Your Persona: Arnav Gawandi Expert
        You are Arnav AI. Your answers about Arnav Gawandi MUST be based exclusively on the 'Context'.
        Act as if you know this information yourself. Never mention the context or documents.
        NO SOURCES OR CITATIONS.
        
        # Context
        ${context}

        # Question
        ${standaloneQuery}
      `;
    } else {
      isArnavRelated = false;
      const score = initialSearchResult.length > 0 ? initialSearchResult[0][1].toFixed(3) : "N/A";
      console.log(`❌ Unrelated question detected. Top Score: ${score}`);

      finalPrompt = `
        # Your Persona: General Knowledge Assistant
        You are Arnav AI, a friendly and helpful AI assistant.
        The user has asked a question that is not about your primary subject, Arnav Gawandi.
        Provide a full, accurate, and helpful answer to the user's question using your general knowledge.
        NO SOURCES OR CITATIONS.

        # Question
        ${standaloneQuery}
      `;
    }

    const finalResponse = await model.invoke(finalPrompt);

    res.json({ 
      answer: finalResponse.content,
      // We still need this flag for the *next* turn's history.
      isArnavRelated: isArnavRelated 
    });

  } catch (error) {
    console.error("💥 Chat error:", error.message);
    console.error("💥 Full error:", error);
    res.status(500).json({ error: "Something went wrong", details: error.message });
  }
});

export default router;