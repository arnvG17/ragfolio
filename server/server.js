import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import dotenv from "dotenv";

import { buildVectorStore } from "./store/memoryStore.js";
import chatRoutes from "./routes/chatRoutes.js";
import simpleChatRoutes from "./routes/simpleChatRoutes.js";

dotenv.config(); // load .env

const PORT = process.env.PORT || 5000;

const app = express();
app.use(cors());
app.use(bodyParser.json());

await buildVectorStore(); // build at startup

app.use("/chat", chatRoutes);
app.use("/simple-chat", simpleChatRoutes);

app.listen(PORT, () => {
  console.log(`🚀 RAG bot running on http://localhost:${PORT}`);
});
