from flask import Flask, request, render_template
from transformers import pipeline
import json
import traceback
import re
import os
from dotenv import load_dotenv
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# OpenAI Configuration - with fallback (If any possible recruiters do not have an OpenAPI key which is loaded with credits 
# please contact me and I can provide one so you can see the API Integration at work, else the fallback (less effective)
# model will be used

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    # Fallback: Set API key directly if not found in .env
    OPENAI_API_KEY = "YourAPIKeyHere"
    print("🔑 Using fallback API key configuration")

# OpenAI client will be created when needed

# Initialize ML components
print("🤖 Initializing AI and ML components...")

# Initialize sentiment analysis
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("✅ Sentiment analyzer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading sentiment analyzer: {e}")
    sentiment_analyzer = None

# Initialize the text generation model (fallback)
print("🔄 Initializing fallback AI model...")
try:
    generator = pipeline(
        "text-generation", 
        model="microsoft/DialoGPT-small",
        device=-1,
        torch_dtype="auto"
    )
    print("✅ Fallback AI model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading fallback model: {e}")
    generator = None

# Simple ML for intent classification
class SimpleIntentClassifier:
    def __init__(self):
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'take care', 'good night'],
            'thanks': ['thank you', 'thanks', 'appreciate it', 'grateful'],
            'help': ['help', 'support', 'assist', 'what can you do', 'how does this work'],
            'weather': ['weather', 'temperature', 'forecast', 'rain', 'sunny'],
            'joke': ['joke', 'funny', 'humor', 'laugh', 'comedy'],
            'fact': ['fact', 'information', 'tell me about', 'what is', 'explain'],
            'opinion': ['what do you think', 'your opinion', 'do you think', 'believe']
        }
        
        # Train simple classifier
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.classifier = MultinomialNB()
        self._train_classifier()
    
    def _train_classifier(self):
        """Train the intent classifier with sample data"""
        texts = []
        labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                texts.append(phrase)
                labels.append(intent)
        
        # Add some variations
        variations = [
            ('greeting', ['hello there', 'hi there', 'hey there', 'good day']),
            ('farewell', ['bye bye', 'see you later', 'take care', 'have a good day']),
            ('thanks', ['thank you so much', 'thanks a lot', 'much appreciated']),
            ('help', ['can you help', 'i need help', 'how to use', 'what are your features']),
            ('weather', ['what\'s the weather', 'weather today', 'is it raining', 'temperature outside']),
            ('joke', ['tell me a joke', 'make me laugh', 'funny story', 'humor me']),
            ('fact', ['give me a fact', 'interesting fact', 'did you know', 'tell me something']),
            ('opinion', ['what\'s your view', 'do you agree', 'your thoughts', 'what\'s your take'])
        ]
        
        for intent, phrases in variations:
            for phrase in phrases:
                texts.append(phrase)
                labels.append(intent)
        
        # Fit the classifier
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        print("✅ Intent classifier trained successfully!")
    
    def classify_intent(self, text):
        """Classify the intent of user input"""
        try:
            X = self.vectorizer.transform([text.lower()])
            intent = self.classifier.predict(X)[0]
            confidence = np.max(self.classifier.predict_proba(X))
            return intent, confidence
        except Exception as e:
            print(f"⚠️ Error in intent classification: {e}")
            return 'general', 0.5

# Initialize intent classifier
intent_classifier = SimpleIntentClassifier()

# User interaction history for personalization
user_history = {}

def get_sentiment(text):
    """Analyze sentiment of user input"""
    try:
        if sentiment_analyzer:
            result = sentiment_analyzer(text)
            return result[0]['label'], result[0]['score']
        return 'NEUTRAL', 0.5
    except Exception as e:
        print(f"⚠️ Error in sentiment analysis: {e}")
        return 'NEUTRAL', 0.5

def get_openai_response(user_input, context=""):
    """Get response from OpenAI API"""
    try:
        if not OPENAI_API_KEY:
            return None, "OpenAI API key not configured"
        
        # Create context-aware prompt
        system_prompt = """You are a helpful, friendly AI assistant. Provide concise, accurate, and engaging responses. 
        If the user asks for help, be supportive. If they share something positive, be encouraging. 
        Keep responses conversational and under 100 words."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        if context:
            messages.insert(1, {"role": "assistant", "content": context})
        
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip(), None
        
    except Exception as e:
        return None, str(e)

def get_fallback_response(user_input):
    """Get response from local model as fallback"""
    try:
        if generator is None:
            return "I'm sorry, but I'm having trouble connecting to my AI brain right now. Please try again later."
        
        prompt = f"Human: {user_input}\nAssistant:"
        
        result = generator(
            prompt, 
            max_new_tokens=50,
            num_return_sequences=1, 
            do_sample=True, 
            temperature=0.7,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.1,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
            early_stopping=True
        )
        
        full_generated_text = result[0]["generated_text"]
        ai_response = extract_ai_response(full_generated_text, user_input)
        
        return ai_response if ai_response else "That's interesting. Tell me more about that."
        
    except Exception as e:
        print(f"❌ Error in fallback response: {e}")
        return "I'm having trouble processing that right now. Could you rephrase your question?"

def extract_ai_response(full_text, user_input):
    """Extract only the AI's response from the generated text"""
    try:
        if full_text.startswith(f"Human: {user_input}"):
            ai_response = full_text[len(f"Human: {user_input}\nAssistant:"):].strip()
        elif full_text.startswith(user_input):
            ai_response = full_text[len(user_input):].strip()
        else:
            ai_response = full_text
        
        ai_response = re.sub(r'<\|endoftext\|>', '', ai_response).strip()
        
        if len(ai_response) < 5:
            return "I understand. How can I help you with that?"
        
        return ai_response if ai_response else "That's interesting. Tell me more about that."
        
    except Exception as e:
        print(f"⚠️ Error extracting AI response: {e}")
        return "I understand your question. Let me help you with that."

def personalize_response(response, user_id, intent, sentiment):
    """Personalize response based on user history and analysis"""
    try:
        # Get user history
        if user_id not in user_history:
            user_history[user_id] = {
                'interactions': [],
                'preferred_intents': {},
                'sentiment_trend': []
            }
        
        # Update user history
        user_history[user_id]['interactions'].append({
            'timestamp': datetime.now(),
            'intent': intent,
            'sentiment': sentiment[0],
            'sentiment_score': sentiment[1]
        })
        
        # Keep only last 10 interactions
        if len(user_history[user_id]['interactions']) > 10:
            user_history[user_id]['interactions'] = user_history[user_id]['interactions'][-10:]
        
        # Analyze user preferences
        recent_interactions = user_history[user_id]['interactions'][-5:]
        if recent_interactions:
            try:
                # Convert sentiment scores to float to avoid dtype issues
                sentiment_scores = [float(i['sentiment_score']) for i in recent_interactions]
                avg_sentiment = np.mean(sentiment_scores)
                
                # Adjust response based on sentiment trend
                if avg_sentiment > 0.6:
                    # User seems positive, be more enthusiastic
                    if not response.startswith(('Great!', 'Excellent!', 'Awesome!')):
                        response = f"Great! {response}"
                elif avg_sentiment < 0.4:
                    # User seems negative, be more supportive
                    if not any(word in response.lower() for word in ['understand', 'support', 'help']):
                        response = f"I understand. {response}"
            except Exception as e:
                print(f"⚠️ Error in sentiment analysis: {e}")
                # Continue without personalization if there's an error
        
        return response
        
    except Exception as e:
        print(f"⚠️ Error in personalization: {e}")
        return response

def get_ai_response(user_input, user_id="default"):
    """Get intelligent AI response with ML enhancements"""
    try:
        print("🧠 Processing user input with ML enhancements...")
        
        # 1. Intent Classification
        intent, confidence = intent_classifier.classify_intent(user_input)
        print(f"🎯 Detected intent: {intent} (confidence: {confidence:.2f})")
        
        # 2. Sentiment Analysis
        sentiment_label, sentiment_score = get_sentiment(user_input)
        print(f"😊 Sentiment: {sentiment_label} (score: {sentiment_score:.2f})")
        
        # 3. Try OpenAI first
        ai_response, error = get_openai_response(user_input)
        
        if ai_response:
            print("✅ Using OpenAI response")
        else:
            print(f"⚠️ OpenAI failed: {error}")
            print("🔄 Falling back to local model...")
            ai_response = get_fallback_response(user_input)
        
        # 4. Personalize response
        personalized_response = personalize_response(ai_response, user_id, intent, sentiment_label)
        
        # 5. Add context based on intent
        if intent == 'greeting':
            if not any(word in personalized_response.lower() for word in ['hello', 'hi', 'hey']):
                personalized_response = f"Hello! {personalized_response}"
        elif intent == 'farewell':
            if not any(word in personalized_response.lower() for word in ['bye', 'goodbye', 'see you']):
                personalized_response = f"{personalized_response} Have a great day!"
        elif intent == 'thanks':
            if not any(word in personalized_response.lower() for word in ['welcome', 'pleasure', 'glad']):
                personalized_response = f"You're welcome! {personalized_response}"
        
        return personalized_response
        
    except Exception as e:
        print(f"❌ Error in AI response generation: {e}")
        return "I'm having trouble processing that right now. Could you rephrase your question?"

@app.route("/")
def index():
    """Serve the main chatbot interface"""
    print("📄 Serving main page (index.html)")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests with ML enhancements"""
    try:
        print("\n" + "="*50)
        print("🤖 New chat request received")
        
        # Get JSON data from request
        request_data = request.get_json()
        print(f"📥 Raw request data: {request_data}")
        
        if not request_data:
            print("❌ No JSON data received")
            return {"error": "No data received"}, 400
        
        # Extract user message and ID
        user_input = request_data.get("message", "")
        user_id = request_data.get("user_id", "default")
        print(f"👤 User message: '{user_input}' (User ID: {user_id})")
        
        if not user_input.strip():
            print("❌ Empty user message")
            return {"error": "Message cannot be empty"}, 400
        
        # Get intelligent AI response with ML enhancements
        ai_response = get_ai_response(user_input, user_id)
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

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get chatbot statistics"""
    try:
        total_users = len(user_history)
        total_interactions = sum(len(user_data['interactions']) for user_data in user_history.values())
        
        stats = {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "active_users": len([u for u in user_history.values() if len(u['interactions']) > 0])
        }
        
        return stats
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {"error": "Could not retrieve statistics"}, 500

if __name__ == "__main__":
    print("🚀 Starting Enhanced AI Chatbot with ML...")
    print("🌐 Server will be available at: http://127.0.0.1:5000")
    print("📝 Debug mode is enabled")
    print("🤖 Features:")
    print("   - OpenAI API integration with local fallback")
    print("   - Intent classification")
    print("   - Sentiment analysis")
    print("   - Response personalization")
    print("   - User interaction tracking")
    print("="*50)
    app.run(debug=True)
