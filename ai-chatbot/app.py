from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import traceback
import re

# Initialize Flask application
app = Flask(__name__)

# Initialize the text generation model
# Using a more reliable model for consistent responses
print("🔄 Initializing AI model (gpt2 - reliable and fast)...")
try:
    # Use GPT-2 which is more reliable for text generation
    generator = pipeline(
        "text-generation", 
        model="gpt2",  
        device=-1,  # Use CPU for faster loading
        torch_dtype="auto"
    )
    print("✅ AI model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading gpt2: {e}")
    print("💡 Trying fallback to distilgpt2...")
    try:
        generator = pipeline(
            "text-generation", 
            model="distilgpt2",
            device=-1,
            torch_dtype="auto"
        )
        print("✅ Fallback model (distilgpt2) loaded successfully!")
    except Exception as e2:
        print(f"❌ Error loading fallback model: {e2}")
        print("💡 Make sure you have internet connection for model download")
        generator = None

def get_smart_response(user_input):
    """Get intelligent responses based on user input patterns"""
    user_input_lower = user_input.lower().strip()
    
    # Greetings
    if user_input_lower in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']:
        return "Hello! How can I help you today?"
    
    # Help requests
    if any(phrase in user_input_lower for phrase in ['help', 'assistance', 'support', 'aid']):
        return "I'd be happy to help you! What would you like assistance with?"
    
    # Questions about the AI
    if any(phrase in user_input_lower for phrase in ['who are you', 'what are you', 'your name', 'what is your name']):
        return "I'm an AI assistant designed to help you with questions and conversations. How can I assist you?"
    
    # How are you
    if 'how are you' in user_input_lower:
        return "I'm functioning well, thank you for asking! How about you?"
    
    # Goodbyes
    if any(phrase in user_input_lower for phrase in ['bye', 'goodbye', 'see you', 'farewell']):
        return "Goodbye! Have a great day!"
    
    # Thank you
    if any(phrase in user_input_lower for phrase in ['thank you', 'thanks', 'appreciate it']):
        return "You're welcome! Is there anything else I can help you with?"
    
    # Weather questions
    if 'weather' in user_input_lower:
        return "I don't have access to real-time weather information, but I'd be happy to help you with other questions!"
    
    # Time questions
    if any(phrase in user_input_lower for phrase in ['what time', 'current time', 'time now']):
        return "I don't have access to real-time information, but I can help you with other questions!"
    
    # General questions
    if user_input_lower.endswith('?'):
        return "That's an interesting question! I'd be happy to help you find information about that topic."
    
    # Statements
    if len(user_input_lower.split()) <= 3:
        return "I understand. How can I help you with that?"
    
    # Default response for complex inputs
    return "I understand what you're saying. How can I assist you with that?"

def extract_ai_response(full_text, user_input):
    """Extract only the AI's response from the generated text"""
    try:
        # For GPT-2, we need to extract the response after the user input
        if full_text.startswith(f"User: {user_input}"):
            # Find where the AI response starts
            ai_start = full_text.find("AI:")
            if ai_start != -1:
                ai_response = full_text[ai_start + 3:].strip()
                # Clean up any remaining patterns
                ai_response = re.sub(r'(User:|AI:).*?$', '', ai_response, flags=re.MULTILINE).strip()
                return ai_response if ai_response else "I'm here to help!"
            else:
                return "I understand. How can I help you?"
        else:
            # If the format is unexpected, provide a default response
            return "I'm here to help! What can I assist you with?"
    except Exception as e:
        print(f"⚠️ Error extracting AI response: {e}")
        return "I'm here to help!"

@app.route("/")
def index():
    """Serve the main chatbot interface"""
    print("📄 Serving main page (index.html)")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests from the frontend"""
    try:
        print("\n" + "="*50)
        print("🤖 New chat request received")
        
        # Get JSON data from request
        request_data = request.get_json()
        print(f"📥 Raw request data: {request_data}")
        
        if not request_data:
            print("❌ No JSON data received")
            return {"error": "No data received"}, 400
        
        # Extract user message
        user_input = request_data.get("message", "")
        print(f"👤 User message: '{user_input}'")
        
        if not user_input.strip():
            print("❌ Empty user message")
            return {"error": "Message cannot be empty"}, 400
        
        # Try smart response first (covers most common cases)
        smart_response = get_smart_response(user_input)
        if smart_response:
            print(f"🧠 Using smart response: '{smart_response}'")
            response_data = {"response": smart_response}
            print(f"📤 Sending response: {response_data}")
            print("="*50 + "\n")
            return response_data
        
        # For very complex queries, use the AI model as fallback
        print("🤖 Using AI model for complex query...")
        
        # Check if model is loaded
        if generator is None:
            print("❌ AI model not available")
            return {"error": "AI model not loaded"}, 500
        
        # Generate AI response for complex queries
        prompt = f"User: {user_input}\nAI:"
        print(f"📝 Using prompt: '{prompt}'")
        
        result = generator(
            prompt, 
            max_new_tokens=20,  # Keep responses short and focused
            num_return_sequences=1, 
            do_sample=True, 
            temperature=0.6,  # Lower temperature for more focused responses
            top_k=30,
            top_p=0.8,
            repetition_penalty=1.1,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True
        )
        print(f"📊 Generation result: {result}")
        
        # Extract the generated text
        full_generated_text = result[0]["generated_text"]
        print(f"🤖 Full generated text: '{full_generated_text}'")
        
        # Extract only the AI's response
        ai_response = extract_ai_response(full_generated_text, user_input)
        print(f"🤖 Cleaned AI response: '{ai_response}'")
        
        # Send response back to frontend
        response_data = {"response": ai_response}
        print(f"📤 Sending response: {response_data}")
        print("="*50 + "\n")
        
        return response_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return {"error": "Invalid JSON format"}, 400
        
    except Exception as e:
        print(f"❌ Unexpected error in chat endpoint: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    print("🚀 Starting Flask AI Chatbot...")
    print("🌐 Server will be available at: http://127.0.0.1:5000")
    print("📝 Debug mode is enabled")
    print("🧠 Using smart response system for reliable answers")
    print("="*50)
    app.run(debug=True)