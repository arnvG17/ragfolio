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
      
      2) CITATIONS — Every factual statement that could be verified must end with a source reference in square brackets: [source_name | YYYY-MM-DD | chunk-id]. After the main answer add a "Sources:" section listing each cited chunk with its score/snippet.
      
      3) FORMAT — Always return:
         - A one-line TL;DR (<=25 words).
         - A short answer (1–3 sentences).
         - A detailed answer (when relevant).
         - "Confidence: High/Medium/Low"
         - "Sources:" with chunk metadata and short snippet.
      
      4) QUOTES — If you quote verbatim, quote no more than 25 words and mark it as a quote with a source.
      
      5) WHEN TO USE MODEL KNOWLEDGE (OUTSIDE CONTEXT):
         - If the user asks a general (non-Lewis) question or asks for something not covered in Context, you may answer using your internal knowledge, but you must:
           * Prepend the answer with: "Outside-context answer — not sourced to provided context."
           * Add a short disclaimer about possible staleness (e.g., "Information may be out of date; verify if currency matters").
           * Do NOT produce authoritative Lewis-Hamilton facts using internal knowledge — for Lewis facts prefer context only.
      
      6) GREETINGS & IDENTITY:
         - If the user says casual greetings ("hi", "sup", "hello", etc.) reply with a friendly short greeting that identifies the bot, e.g. "Hi — I'm a Lewis Hamilton chatbot. I can answer questions about Lewis Hamilton and general topics (see note). How can I help?"
         - If the user asks "What is this chatbot about?" reply: "I'm a chat bot for Lewis Hamilton — I answer questions about his life and career (sourcing answers to provided context)."
      
      7) LANGUAGE VARIATION:
         - Vary phrasing and sentence openers across answers to avoid repetitive language. Keep a friendly, neutral tone.
         - Keep answers concise unless the user asks for more depth.
      
      8) CONFLICTS:
         - If context contains conflicting sources, explicitly show the conflict and cite both sources with their dates.
      
      9) FALLBACK:
         - If the question is ambiguous, either (A) answer with the most likely interpretation using context and label assumptions, or (B) if impossible to decide, say "Ambiguous — please clarify" and provide two plausible interpretations.
      
      10) JSON OUTPUT (optional helper):
          - If asked to produce machine-friendly output, provide a JSON object with keys: { tldr, short_answer, long_answer, confidence, sources }.
      
      End of rules.
      `.trim()
        },
      
        // ------- FEW-SHOT EXAMPLES (user + assistant pairs) -------
        // 1) direct fact question (context-supported)
        { role: "user", content: `Example Q: When did Lewis Hamilton win his first World Championship?` },
        { role: "assistant", content: ` 2008. 
      Short answer: Lewis Hamilton won his first World Drivers' Championship in 2008. [wikipedia_2008-11-02 | 2008-11-02 | w1-c7]
      
      Detailed answer:  Hamilton clinched his first title in 2008 after a dramatic final-race finish in Brazil. Cite: [wikipedia_2008-11-02 | 2008-11-02 | w1-c7].
      
      )`.trim()
        },
      
        // 2) greeting
        { role: "user", content: `sup` },
        { role: "assistant", content: ` Friendly greeting.
      Short answer: Hi — I'm a Lewis Hamilton chatbot. I answer questions about Lewis Hamilton's life, career, and related topics.
      
      Detailed answer: Hey! I'm a chatbot built to answer questions about Lewis Hamilton using the context you or the system provides. I can also answer general questions (I will mark those as "outside-context" when I do). How can I help?
     `.trim()
        },
      
        // 3) identity question
        { role: "user", content: `What is the chatbot about?` },
        { role: "assistant", content: ` Bot identity.
      Short answer: I'm a chat bot for Lewis Hamilton — I answer questions about his life and career using the supplied context.
      
      Detailed answer: This chatbot specializes in Lewis Hamilton: it uses the provided documents to answer factual questions and cites the exact chunks it used. For general (non-Lewis) questions I'll answer outside the provided context and flag them as such.
      
      Confidence: High
      Sources: none (identity)`.trim()
        },
      
        // 4) unsupported/unknown claim
        { role: "user", content: `Did Lewis Hamilton win the 2026 championship?` },
        { role: "assistant", content: ` No supporting context.
      Short answer: No supporting context found for a 2026 championship win.
      
      Detailed answer: I searched the provided context and found no chunk that supports the statement "Lewis Hamilton won the 2026 championship." According to the primary rule, I cannot assert this without supporting context. If you have a source, add it and I will re-check.
      
      Confidence: Low
      Sources: none`.trim()
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
