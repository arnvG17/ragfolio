import fetch from 'node-fetch';

async function testGreeting() {
  try {
    console.log("🧪 Testing simple greeting...");
    
    const response = await fetch('http://localhost:5000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: "hi" })
    });
    
    const data = await response.json();
    
    if (response.ok) {
      console.log("✅ Response:", data.answer);
    } else {
      console.log("❌ Error:", data.error);
    }
  } catch (error) {
    console.log("💥 Error:", error.message);
  }
}

testGreeting();
