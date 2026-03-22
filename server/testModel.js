import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { config as configDotenv } from "dotenv";

configDotenv();

async function test() {
  try {
    console.log("🧪 Testing chat model...");
    const model = new ChatGoogleGenerativeAI({
      model: process.env.GOOGLE_MODEL || "gemini-2.5-flash",
      apiKey: process.env.GOOGLE_API_KEY,
    });
    
    console.log("✅ Model initialized");
    
    const response = await model.invoke("Hello, who are you?");
    console.log("✅ Model response:", response.content);
  } catch (error) {
    console.error("❌ Error:", error.message);
    console.error("❌ Full error:", error);
  }
}

test();
