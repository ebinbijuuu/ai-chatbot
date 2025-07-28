from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import traceback
import re

# Initialize Flask application
app = Flask(__name__)

# Initialize the text generation model
# Using a faster model while maintaining quality
print("🔄 Initializing AI model (microsoft/DialoGPT-small - fast & intelligent)...")
try:
    # Use DialoGPT-small for faster responses while maintaining good quality
    generator = pipeline(
        "text-generation", 
        model="microsoft/DialoGPT-small",  # Faster than medium, still good quality
        device=-1,  # Use CPU for faster loading
        torch_dtype="auto"
    )
    print("✅ AI model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading DialoGPT-small: {e}")
    print("💡 Trying fallback to distilgpt2...")
    try:
        generator = pipeline(
            "text-generation", 
            model="distilgpt2",  # Even faster fallback
            device=-1,
            torch_dtype="auto"
        )
        print("✅ Fallback model (distilgpt2) loaded successfully!")
    except Exception as e2:
        print(f"❌ Error loading fallback model: {e2}")
        print("💡 Make sure you have internet connection for model download")
        generator = None

# Simple response cache for common questions
response_cache = {}

def get_ai_response(user_input):
    """Get intelligent AI response for any type of question - optimized for speed"""
    try:
        # Check if model is loaded
        if generator is None:
            return "I'm sorry, but I'm having trouble connecting to my AI brain right now. Please try again later."
        
        # Check cache first for speed
        user_input_lower = user_input.lower().strip()
        if user_input_lower in response_cache:
            print("⚡ Using cached response for speed")
            return response_cache[user_input_lower]
        
        print("🧠 Generating intelligent AI response...")
        
        # Use different prompting strategies based on the model
        model_name = str(generator.model.config._name_or_path)
        
        if "microsoft/DialoGPT" in model_name:
            # DialoGPT works better with conversational prompts
            prompt = f"Human: {user_input}\nAssistant:"
        else:
            # For other models, use conversation format
            prompt = f"User: {user_input}\nAI:"
            
        print(f"📝 Using prompt: '{prompt}'")
        
        # Optimized parameters for speed while maintaining quality
        result = generator(
            prompt, 
            max_new_tokens=35,  # Reduced for speed, still good length
            num_return_sequences=1, 
            do_sample=True, 
            temperature=0.7,  # Balanced for speed and creativity
            top_k=30,  # Reduced for speed
            top_p=0.85,  # Slightly reduced for speed
            repetition_penalty=1.1,  # Reduced for speed
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
            early_stopping=True  # Stop generation early if possible
        )
        
        print(f"📊 Generation result: {result}")
        
        # Extract the generated text
        full_generated_text = result[0]["generated_text"]
        print(f"🤖 Full generated text: '{full_generated_text}'")
        
        # Extract only the AI's response
        ai_response = extract_ai_response(full_generated_text, user_input, model_name)
        print(f"🤖 Cleaned AI response: '{ai_response}'")
        
        # Cache the response for future use (limit cache size)
        if len(response_cache) < 50:  # Keep cache manageable
            response_cache[user_input_lower] = ai_response
        
        return ai_response
        
    except Exception as e:
        print(f"❌ Error generating AI response: {e}")
        return "I'm having trouble processing that right now. Could you rephrase your question?"

def extract_ai_response(full_text, user_input, model_name):
    """Extract only the AI's response from the generated text - optimized for speed"""
    try:
        if "microsoft/DialoGPT" in model_name:
            # For DialoGPT models - optimized extraction
            if full_text.startswith(f"Human: {user_input}"):
                # Remove the "Human: ... Assistant:" format
                ai_response = full_text[len(f"Human: {user_input}\nAssistant:"):].strip()
            elif full_text.startswith(user_input):
                # Remove the user input from the beginning
                ai_response = full_text[len(user_input):].strip()
            else:
                # If format is unexpected, use the full text
                ai_response = full_text
            
            # Quick cleanup
            ai_response = re.sub(r'<\|endoftext\|>', '', ai_response).strip()
            
        else:
            # For other models - optimized extraction
            if full_text.startswith(f"User: {user_input}"):
                ai_start = full_text.find("AI:")
                if ai_start != -1:
                    ai_response = full_text[ai_start + 3:].strip()
                else:
                    ai_response = full_text[len(f"User: {user_input}"):].strip()
            else:
                ai_response = full_text
        
        # Quick cleanup
        ai_response = re.sub(r'<\|endoftext\|>', '', ai_response).strip()
        
        # If response is too short, provide a quick fallback
        if len(ai_response) < 5:
            return "I understand. How can I help you with that?"
        
        return ai_response if ai_response else "That's interesting. Tell me more about that."
        
    except Exception as e:
        print(f"⚠️ Error extracting AI response: {e}")
        return "I understand your question. Let me help you with that."

@app.route("/")
def index():
    """Serve the main chatbot interface"""
    print("📄 Serving main page (index.html)")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests from the frontend - optimized for speed"""
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
        
        # Get intelligent AI response for any type of question
        ai_response = get_ai_response(user_input)
        print(f"🧠 AI Response: '{ai_response}'")
        
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
    print("⚡ Optimized for speed with DialoGPT-small")
    print("="*50)
    app.run(debug=True)
