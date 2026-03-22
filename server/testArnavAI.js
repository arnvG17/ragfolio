import fetch from 'node-fetch';

const testQuestions = [
  "What college does Arnav attend?"
];

async function testArnavAI() {
  console.log('🧪 Testing Arnav AI...\n');
  
  for (const question of testQuestions) {
    try {
      console.log(`❓ Question: "${question}"`);
      
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question })
      });
      
      console.log(`📊 Response status: ${response.status}`);
      
      const data = await response.text();
      console.log(`📄 Raw response: ${data}`);
      
      const jsonData = JSON.parse(data);
      
      if (response.ok) {
        console.log(`✅ Answer: "${jsonData.answer}"`);
        console.log(`📊 Arnav-related: ${jsonData.isArnavRelated ? 'Yes' : 'No'}\n`);
      } else {
        console.log(`❌ Error: ${jsonData.error}\n`);
      }
    } catch (error) {
      console.log(`💥 Error: ${error.message}\n`);
    }
  }
}

testArnavAI();
