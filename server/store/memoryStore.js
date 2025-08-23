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

    // Re-add docs
    for (const item of serialized) {
      await vectorStore.addDocuments([
        {
          pageContent: item.pageContent,
          metadata: item.metadata,
        },
      ]);
    }

    chunksIndexed = serialized.length;
  } else {
    throw new Error("vectorStore.json missing! Run precompute.js first.");
  }

  return { chunksIndexed };
}

export function getVectorStore() {
  return vectorStore;
}

export function getChunksIndexed() {
  return chunksIndexed;
}
