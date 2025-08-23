// precompute.js
import fs from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { embeddings } from "./embeddings/geminiEmbeddings.js";

async function main() {
  const raw = fs.readFileSync("me.txt", "utf-8");

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 550,
    chunkOverlap: 100,
  });

  const docs = await splitter.createDocuments([raw]);
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  // Instead of reading internals, just serialize docs we already have
  const serialized = [];
  for (const doc of docs) {
    serialized.push({
      pageContent: doc.pageContent,
      metadata: doc.metadata,
      embedding: await embeddings.embedQuery(doc.pageContent),
    });
  }

  fs.writeFileSync("vectorStore.json", JSON.stringify(serialized, null, 2));
  console.log("✅ Vector store saved to vectorStore.json");
}

main();
