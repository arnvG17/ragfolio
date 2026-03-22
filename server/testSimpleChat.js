import fetch from 'node-fetch';

async function testSimpleChat() {
  try {
    console.log("🧪 Testing simple chat endpoint...");
    
    const response = await fetch('http://localhost:5000/simple-chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: "What college does Arnav attend?" })
    });
    
    const data = await response.json();
    
    if (response.ok) {
      console.log("✅ Answer:", data.answer);
      console.log("📊 Score:", data.score);
      console.log("🤖 Arnav-related:", data.isArnavRelated);
    } else {
      console.log("❌ Error:", data.error);
    }
  } catch (error) {
    console.log("💥 Error:", error.message);
  }
}

testSimpleChat();
