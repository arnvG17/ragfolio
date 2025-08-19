import express from "express";
import cors from "cors";
import bodyParser from "body-parser";

import { buildVectorStore } from "./store/memoryStore.js";
import chatRoutes from "./routes/chatRoutes.js";
import { PORT } from "./config/env.js";

const app = express();
app.use(cors());
app.use(bodyParser.json());

await buildVectorStore(); // build at startup

app.use("/chat", chatRoutes);


app.listen(PORT, () => {
  console.log(` RAG bot running on http://localhost:${PORT}`);
});
