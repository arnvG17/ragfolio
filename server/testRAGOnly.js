import fetch from 'node-fetch';

async function testRAGOnly() {
  try {
    console.log("🧪 Testing RAG system only...");
    
    // Test with a question that should work
    const response = await fetch('http://localhost:5000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: "What college does Arnav attend?" })
    });
    
    const data = await response.text();
    console.log("📄 Raw response:", data);
    
  } catch (error) {
    console.log("💥 Error:", error.message);
  }
}

testRAGOnly();
