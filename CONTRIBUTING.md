# 🤝 Contributing to RAGfolio

Welcome to **RAGfolio** – an AI-powered portfolio with a built-in Retrieval-Augmented Generation (RAG) chatbot! This document will help you understand the project architecture, how the embedding system works, how prompts are designed, and how you can contribute.

---

## 📚 Table of Contents

1. [Project Overview](#-project-overview)
2. [Tech Stack](#-tech-stack)
3. [Architecture Diagram](#-architecture-diagram)
4. [Backend Deep Dive](#-backend-deep-dive)
   - [Directory Structure](#directory-structure)
   - [Server Entry Point](#server-entry-point)
   - [Embeddings System](#embeddings-system)
   - [Vector Store](#vector-store)
   - [Chat Routes & Prompt Design](#chat-routes--prompt-design)
5. [Frontend Overview](#-frontend-overview)
6. [How to Contribute](#-how-to-contribute)
7. [Setting Up the Development Environment](#-setting-up-the-development-environment)
8. [Code Style Guidelines](#-code-style-guidelines)
9. [Submitting Changes](#-submitting-changes)

---

## 🎯 Project Overview

**RAGfolio** is a personal portfolio website built with React (Frontend) and Node.js (Backend) that features a **RAG-powered chatbot**. The chatbot uses semantic search to find relevant information from a knowledge base and generates contextual responses using Google's Gemini AI.

### Key Features
- 🤖 **RAG Chatbot**: Retrieves relevant context from a vector store before generating responses
- 🔄 **Easy Customization**: Simply replace `me.txt` with your own content and regenerate embeddings
- 🧠 **Conversation Memory**: Maintains chat history for contextual follow-up questions
- ⚡ **Semantic Search**: Uses cosine similarity for accurate document retrieval

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Node.js + Express** | Server runtime and HTTP framework |
| **LangChain** | Text splitting, embeddings, and vector store utilities |
| **Google Gemini AI** | LLM for chat responses and text embeddings |
| **dotenv** | Environment variable management |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React + Vite** | UI framework and build tool |
| **TailwindCSS** | Styling |
| **Framer Motion** | Animations |
| **Lucide React** | Icons |

---

## 🏗 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐                      │
│  │   ChatWidget    │───▶│     Chatbot     │                      │
│  │  (Floating UI)  │    │   (Messages)    │                      │
│  └─────────────────┘    └────────┬────────┘                      │
│                                  │ POST /chat                    │
└──────────────────────────────────┼───────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND (Node.js)                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                        server.js                             │ │
│  │  • CORS, Body Parser                                        │ │
│  │  • Loads Vector Store at Startup                            │ │
│  │  • Routes: /chat                                            │ │
│  └───────────────────────────┬─────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────▼─────────────────────────────────┐ │
│  │                     chatRoutes.js                            │ │
│  │                                                              │ │
│  │  1. Simple Route Matching (greetings)                       │ │
│  │  2. Query Condensation (with chat history)                  │ │
│  │  3. Vector Search (MMR or Similarity)                       │ │
│  │  4. Relevance Check (threshold: 0.69)                       │ │
│  │  5. Context-Aware Prompt Construction                       │ │
│  │  6. Gemini Response Generation                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │ geminiEmbeddings│    │  memoryStore    │    │vectorStore-  │  │
│  │      .js        │    │      .js        │    │  Loader.js   │  │
│  │                 │    │                 │    │              │  │
│  │ embedding-001   │    │ Loads from JSON │    │ Cosine Sim   │  │
│  │ Google Embeddings│   │ at server start │    │ Search API   │  │
│  └─────────────────┘    └─────────────────┘    └──────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                     precompute.js                            │ │
│  │                                                              │ │
│  │  Offline Script: me.txt → chunks → embeddings → JSON        │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Backend Deep Dive

### Directory Structure

```
backend/server/
├── .env                    # Environment variables (API keys)
├── me.txt                  # Source knowledge file
├── vectorStore.json        # Precomputed embeddings
├── server.js               # Express app entry point
├── precompute.js           # Embedding generation script
├── package.json            # Dependencies
│
├── embeddings/
│   └── geminiembeddings.js # Google AI Embeddings config
│
├── routes/
│   └── chatRoutes.js       # Main chat API logic
│
├── store/
│   └── memoryStore.js      # Vector store builder
│
└── utils/
    └── vectorStoreLoader.js # Custom vector search implementation
```

---

### Server Entry Point

**File: `server.js`**

```javascript
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import dotenv from "dotenv";
import { buildVectorStore } from "./store/memoryStore.js";
import chatRoutes from "./routes/chatRoutes.js";

dotenv.config(); // Load .env

const PORT = process.env.PORT || 5500;

const app = express();
app.use(cors());
app.use(bodyParser.json());

await buildVectorStore(); // Build vector store at startup

app.use("/chat", chatRoutes);

app.listen(PORT, () => {
  console.log(`🚀 RAG bot running on http://localhost:${PORT}`);
});
```

**Key Points:**
- Loads environment variables from `.env`
- Initializes the vector store **before** starting the server
- Exposes a single `/chat` endpoint for chatbot interactions

---

### Embeddings System

#### How Embeddings Work

Embeddings convert text into numerical vectors that capture semantic meaning. Similar texts have similar vectors, enabling semantic search.

**File: `embeddings/geminiembeddings.js`**

```javascript
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv();

export const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: "embedding-001", // Optimized for vector search
  apiKey: process.env.GOOGLE_API_KEY,
});
```

**The `embedding-001` model** is Google's specialized model for generating high-quality embeddings suitable for semantic similarity search.

---

#### Precomputing Embeddings

**File: `precompute.js`**

This script runs **offline** to generate embeddings from your knowledge base.

```javascript
import fs from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { embeddings } from "./embeddings/geminiEmbeddings.js";

async function main() {
  // 1. Read the source knowledge file
  const raw = fs.readFileSync("me.txt", "utf-8");

  // 2. Split into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 350,      // Max characters per chunk
    chunkOverlap: 70,    // Overlap for context continuity
  });

  const docs = await splitter.createDocuments([raw]);
  
  // 3. Generate embeddings for each chunk
  const serialized = [];
  for (const doc of docs) {
    serialized.push({
      pageContent: doc.pageContent,
      metadata: doc.metadata,
      embedding: await embeddings.embedQuery(doc.pageContent),
    });
  }

  // 4. Save to JSON
  fs.writeFileSync("vectorStore.json", JSON.stringify(serialized, null, 2));
  console.log("✅ Vector store saved to vectorStore.json");
}

main();
```

**Chunking Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunkSize` | 350 | Maximum characters per document chunk |
| `chunkOverlap` | 70 | Overlap between consecutive chunks (maintains context) |

---

### Vector Store

#### Loading the Vector Store

**File: `store/memoryStore.js`**

```javascript
import fs from "fs";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { embeddings } from "../embeddings/geminiembeddings.js";

let vectorStore;
let chunksIndexed = 0;

export async function buildVectorStore() {
  if (fs.existsSync("./vectorStore.json")) {
    console.log("📥 Loading precomputed vector store...");
    const serialized = JSON.parse(fs.readFileSync("./vectorStore.json", "utf-8"));

    vectorStore = new MemoryVectorStore(embeddings);

    // Re-add documents
    for (const item of serialized) {
      await vectorStore.addDocuments([{
        pageContent: item.pageContent,
        metadata: item.metadata,
      }]);
    }

    chunksIndexed = serialized.length;
  } else {
    throw new Error("vectorStore.json missing! Run precompute.js first.");
  }

  return { chunksIndexed };
}
```

#### Custom Vector Search

**File: `utils/vectorStoreLoader.js`**

This file implements custom search methods using **cosine similarity**:

```javascript
// Cosine similarity for scoring documents
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

**Search Methods:**
| Method | Description |
|--------|-------------|
| `similaritySearch(query, k)` | Returns top-k most similar documents |
| `maxMarginalRelevanceSearch(query, {k, fetchK})` | Returns documents with score tuples |
| `similaritySearchWithScore(query, k)` | Returns documents with their similarity scores |

---

### Chat Routes & Prompt Design

**File: `routes/chatRoutes.js`**

This is the heart of the RAG chatbot logic.

#### Request Flow

```
User Query
    │
    ▼
┌──────────────────────┐
│ 1. Simple Matching   │  ──▶ Quick responses for greetings
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. Query Condensing  │  ──▶ Resolves pronouns using chat history
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 3. Vector Search     │  ──▶ Finds relevant documents
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 4. Relevance Check   │  ──▶ Score threshold: 0.69
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌─────────────┐
│Related │   │ Unrelated   │
└───┬────┘   └──────┬──────┘
    │               │
    ▼               ▼
  RAG Prompt    General Prompt
    │               │
    └───────┬───────┘
            ▼
     Gemini Response
```

#### Prompt Templates

**1. Query Condensation Prompt**

When the user asks a follow-up question with pronouns (e.g., "What team was he with?"), this prompt resolves them:

```javascript
const condensingPrompt = [
  new HumanMessage(`Given the chat history and a follow-up question, 
  rephrase the follow-up question into a single, standalone question 
  that can be understood without the chat history.
  
  Chat History:
  human: What year did Lewis Hamilton win his first championship?
  ai: He won his first World Championship in 2008.
  
  Follow-Up Question: What team was he with?
  
  Standalone question:`),
  new AIMessage("What team was Lewis Hamilton with when he won his first championship in 2008?"),
  // ... current conversation context
];
```

**2. RAG Prompt (Hamilton-Related Questions)**

When the query is relevant to the knowledge base (score ≥ 0.69):

```javascript
finalPrompt = `
  # Your Persona: Lewis Hamilton Expert
  You are Hamilton AI. Your answers about Lewis Hamilton MUST be based 
  exclusively on the 'Context'. Act as if you know this information yourself.
  Never mention the context or documents.
  NO SOURCES OR CITATIONS.
  
  # Context
  ${context}

  # Question
  ${standaloneQuery}
`;
```

**3. General Knowledge Prompt (Unrelated Questions)**

When the query is not related to the knowledge base:

```javascript
finalPrompt = `
  # Your Persona: General Knowledge Assistant
  You are Hamilton AI, a friendly and helpful AI assistant.
  The user has asked a question that is not about your primary subject.
  Provide a full, accurate, and helpful answer using your general knowledge.
  NO SOURCES OR CITATIONS.

  # Question
  ${standaloneQuery}
`;
```

#### Relevance Threshold

```javascript
const RELEVANCE_THRESHOLD = 0.69;

if (initialSearchResult.length > 0 && initialSearchResult[0][1] >= RELEVANCE_THRESHOLD) {
  isHamiltonRelated = true;
  // Use RAG with context
} else {
  isHamiltonRelated = false;
  // Use general knowledge
}
```

---

## 🎨 Frontend Overview

The frontend uses React with a floating chat widget.

**File: `Frontend/src/components/ChatWidget.jsx`**

```jsx
export default function ChatWidget() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Floating bubble button */}
      <button onClick={() => setOpen(prev => !prev)}>
        {open ? <X /> : <MessageCircle />}
      </button>

      {/* Chatbot panel */}
      {open && (
        <div>
          <Chatbot
            apiUrl="https://ragfolio-1.onrender.com/chat"
            title="Ask Arnav"
          />
        </div>
      )}
    </>
  );
}
```

---

## 🚀 How to Contribute

We welcome contributions of all kinds! Here's how you can help:

### Types of Contributions

| Type | Description |
|------|-------------|
| 🐛 **Bug Fixes** | Fix issues in the codebase |
| ✨ **Features** | Add new functionality |
| 📚 **Documentation** | Improve docs, add examples |
| 🧪 **Testing** | Add or improve tests |
| 🎨 **UI/UX** | Enhance the frontend design |
| ⚡ **Performance** | Optimize existing code |

### Good First Issues

- Improve error handling in `chatRoutes.js`
- Add loading states to the chat widget
- Implement message timestamps
- Add conversation export functionality
- Improve mobile responsiveness

---

## 💻 Setting Up the Development Environment

### Prerequisites

- **Node.js** v18+
- **npm** or **yarn**
- **Google AI API Key** (for Gemini access)

### Backend Setup

```bash
# Navigate to backend
cd backend/server

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Add your API keys to .env
# GOOGLE_API_KEY=your_embedding_api_key
# GOOGLE_API_KEY2=your_gemini_chat_api_key

# Generate vector store (if you've modified me.txt)
node precompute.js

# Start the server
npm start
```

### Frontend Setup

```bash
# Navigate to frontend
cd Frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Customizing the Knowledge Base

1. Edit `backend/server/me.txt` with your content
2. Run `node precompute.js` to regenerate embeddings
3. The new `vectorStore.json` will be created
4. Restart the server

---

## 📝 Code Style Guidelines

### JavaScript/TypeScript

- Use **ES6+ syntax** (arrow functions, destructuring, etc.)
- Prefer `const` over `let`; avoid `var`
- Use meaningful variable names
- Add comments for complex logic

### Git Commit Messages

Follow conventional commits:

```
feat: add new feature
fix: bug fix
docs: documentation changes
style: formatting changes
refactor: code restructuring
test: adding tests
chore: maintenance tasks
```

### Pull Request Guidelines

1. Keep PRs focused on a single change
2. Update documentation if needed
3. Test your changes locally
4. Provide a clear description of changes

---

## 📤 Submitting Changes

### Workflow

1. **Fork** the repository
2. **Clone** your fork locally
3. Create a **feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes
5. **Commit** with descriptive messages
6. **Push** to your fork
7. Open a **Pull Request**

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All existing tests pass
- [ ] New functionality has documentation
- [ ] Commit messages are descriptive
- [ ] PR description explains the changes

---

## 📞 Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

---

## 📄 License

This project is open source. See the LICENSE file for details.

---

**Thank you for contributing to RAGfolio!** 🎉

Your contributions help make this project better for everyone.
