import express from "express";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";

const router = express.Router();

console.log("🔧 Simple chat routes loaded");

// Simple RAG endpoint without chat model
router.post("/", async (req, res) => {
  console.log("🚀 Simple chat route hit");
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    console.log(`🔍 Processing query: "${query}"`);

    const vectorStore = await loadVectorStore();
    const initialSearchResult = await vectorStore.maxMarginalRelevanceSearch(query, {
      k: 1,
      fetchK: 5,
    });
    
    const RELEVANCE_THRESHOLD = 0.69;
    
    if (initialSearchResult.length > 0 && initialSearchResult[0][1] >= RELEVANCE_THRESHOLD) {
      console.log(`✅ Arnav-related question detected. Score: ${initialSearchResult[0][1].toFixed(3)}`);
      
      const contextResults = await vectorStore.similaritySearch(query, 4);
      const context = contextResults
        .map((r, i) => `--- Document ${i + 1} ---\n${r.pageContent}`)
        .join("\n\n");
      
      // Simple answer extraction
      let answer = "I found information about Arnav: ";
      
      if (query.toLowerCase().includes("college")) {
        const collegeMatch = context.match(/Institution: ([^\n]+)/);
        if (collegeMatch) {
          answer += `Arnav attends ${collegeMatch[1]}.`;
        }
      } else if (query.toLowerCase().includes("study") || query.toLowerCase().includes("degree")) {
        const degreeMatch = context.match(/Degree: ([^\n]+)/);
        if (degreeMatch) {
          answer += `Arnav is studying ${degreeMatch[1]}.`;
        }
      } else if (query.toLowerCase().includes("club") || query.toLowerCase().includes("technical")) {
        const clubMatch = context.match(/DJS S4DS.*?DJS CODEAI/);
        if (clubMatch) {
          answer += `Arnav is part of ${clubMatch[0]} clubs.`;
        }
      } else {
        answer += "I found relevant information in Arnav's profile.";
      }
      
      res.json({ 
        answer: answer,
        context: context.substring(0, 200) + "...",
        isArnavRelated: true,
        score: initialSearchResult[0][1]
      });
    } else {
      const score = initialSearchResult.length > 0 ? initialSearchResult[0][1].toFixed(3) : "N/A";
      console.log(`❌ Unrelated question detected. Top Score: ${score}`);
      
      res.json({ 
        answer: "I can only answer questions about Arnav Gawandi. Please ask about his education, experience, or interests.",
        isArnavRelated: false,
        score: score
      });
    }

  } catch (error) {
    console.error("💥 Simple RAG error:", error.message);
    res.status(500).json({ error: "Something went wrong", details: error.message });
  }
});

export default router;
