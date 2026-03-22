import { loadVectorStore } from "./utils/vectorStoreLoader.js";

async function test() {
  try {
    console.log("🧪 Testing vector store loading...");
    const vectorStore = await loadVectorStore();
    console.log("✅ Vector store loaded successfully");
    
    console.log("🧪 Testing search...");
    const result = await vectorStore.maxMarginalRelevanceSearch("What college does Arnav attend?", { k: 1, fetchK: 5 });
    console.log("✅ Search successful:", result.length, "results");
  } catch (error) {
    console.error("❌ Error:", error.message);
    console.error("❌ Full error:", error);
  }
}

test();
