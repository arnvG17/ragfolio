import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { loadVectorStore } from "../utils/vectorStoreLoader.js";
import { config as configDotenv } from "dotenv";

configDotenv();

const router = express.Router();

const model = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_MODEL || "gemini-1.5-flash",
  apiKey: process.env.GOOGLE_API_KEY, // ✅ now from .env
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

      const response = await model.invoke([
        {
          role: "system",
          content: `
      You are a focused RAG assistant for questions about Lewis Hamilton. Follow these rules strictly:
      
      1) PRIMARY RULE — Use only the provided Context for any factual claims about Lewis Hamilton. If a factual claim is not supported by the context, answer: "No supporting context found." Do not hallucinate facts.
      
      
      3) FORMAT — Always return:
         make a human like conversation
         mention whenever you use data or information out of the context of the ducument, like if the user ask--{"what is blackhole ," or what is drag reduction system DRS"}
         
         
         
         5) WHEN TO USE MODEL KNOWLEDGE (OUTSIDE CONTEXT):
         - If the user asks a general (non-Lewis) question or asks for something not covered in Context, you may answer using your internal knowledge, but you must:
            mention whenever you use data or information out of the context of the ducument, like if the user ask--{"what is blackhole ," or what is drag reduction system DRS"}
           * Prepend the answer with: "Outside-context answer — not sourced to provided context."
           * Add a short disclaimer about possible staleness (e.g., "Information may be out of date; verify if currency matters").
           * Do NOT produce authoritative Lewis-Hamilton facts using internal knowledge — for Lewis facts prefer context only.
           * use you own powers to answer those
      
      6) GREETINGS & IDENTITY:
         - If the user says casual greetings ("hi", "sup", "hello", etc.) reply with a friendly short greeting that identifies the bot, e.g. "Hi — I'm a Lewis Hamilton chatbot. I can answer questions about Lewis Hamilton and general topics (see note). How can I help?"
         - If the user asks "What is this chatbot about?" reply: "I'm a chat bot for Lewis Hamilton — I answer questions about his life and career (sourcing answers to provided context)."
      
      7) LANGUAGE VARIATION:
         - Vary phrasing and sentence openers across answers to avoid repetitive language. Keep a friendly, neutral tone.
        
      
      
      End of rules.
      `.trim()
        },
      
        // ------- FEW-SHOT EXAMPLES (user + assistant pairs) -------
        // 1) direct fact question (context-supported)
        { role: "user", content: `Example Q: When did Lewis Hamilton win his first World Championship?` },
        { role: "assistant", content: ` 2008. 
      
      
       Hamilton clinched his first title in 2008 after a dramatic final-race finish in Brazil. Cite: [wikipedia_2008-11-02 | 2008-11-02 | w1-c7].
      
      )`.trim()
        },
      
        // 2) greeting
        { role: "user", content: `sup` },
        { role: "assistant", content: ` Friendly greeting.
      .
      
      Hey! I'm a chatbot built to answer questions about Lewis Hamilton using the context you or the system provides. I can also answer general questions (I will mark those as "outside-context" when I do). How can I help?
     `.trim()
        },
      
        // 3) identity question
        { role: "user", content: `What is the chatbot about?` },
        { role: "assistant", content: ` Bot identity.
      S
      
      This chatbot specializes in Lewis Hamilton: it uses the provided documents to answer factual questions and cites the exact chunks it used. For general (non-Lewis) questions I'll answer outside the provided context and flag them as such.
      
      )`.trim()
        },
      
        // 4) unsupported/unknown claim
        { role: "user", content: `what is drag reduction system` },
        { role: "assistant", content: ` No supporting context.
      
      
      I searched the provided context and found no chunk that supports the statement but heres an explanation- use your powers, use MODEL KNOWLEDGE (OUTSIDE CONTEXT): 
      `.trim()
        },
      
        // ------- END FEW-SHOT -------
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${query}` }
      ]);
      

    res.json({ answer: response.content, sources: results });
  } catch (error) {
    console.error("💥 Chat error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

export default router;
