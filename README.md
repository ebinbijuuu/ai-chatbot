# 🤖 Enhanced AI Chatbot with Machine Learning

A sophisticated AI chatbot that combines OpenAI's GPT models with local fallback and advanced machine learning features for intelligent conversation.

## ✨ Features

### 🧠 AI Models
- **OpenAI GPT-3.5-turbo**: Primary AI model for high-quality responses
- **Local Fallback**: DialoGPT-small for offline functionality
- **Automatic Fallback**: Seamlessly switches between models

### 🎯 Machine Learning Capabilities
- **Intent Classification**: Understands user intentions (greeting, farewell, help, etc.)
- **Sentiment Analysis**: Analyzes emotional tone of messages
- **Response Personalization**: Adapts responses based on user history and preferences
- **User Interaction Tracking**: Maintains conversation history for better context

### 🎨 Modern UI
- **Beautiful Interface**: Gradient backgrounds with glassmorphism effects
- **Real-time Feedback**: Shows ML processing status
- **Statistics Dashboard**: Displays usage statistics
- **Responsive Design**: Works on all devices

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API (Optional)
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Chatbot
Open your browser and go to: `http://127.0.0.1:5000`

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `DEBUG`: Set to 'true' for debug mode

### Model Configuration
The chatbot automatically uses the best available model:
1. **OpenAI GPT-3.5-turbo** (if API key is provided)
2. **DialoGPT-small** (local fallback)
3. **DistilGPT2** (emergency fallback)

## 🧠 Machine Learning Features

### Intent Classification
The chatbot can recognize 8 different user intents:
- **Greeting**: "Hello", "Hi", "Good morning"
- **Farewell**: "Bye", "Goodbye", "See you"
- **Thanks**: "Thank you", "Thanks", "Appreciate it"
- **Help**: "Help", "Support", "What can you do"
- **Weather**: "Weather", "Temperature", "Forecast"
- **Joke**: "Tell me a joke", "Make me laugh"
- **Fact**: "Give me a fact", "Tell me about"
- **Opinion**: "What do you think", "Your opinion"

### Sentiment Analysis
- Uses Twitter RoBERTa model for sentiment analysis
- Categorizes messages as Positive, Negative, or Neutral
- Adjusts response tone based on sentiment

### Personalization
- Tracks user interaction history
- Adapts responses based on user preferences
- Maintains conversation context
- Provides personalized greetings and responses

## 📊 API Endpoints

### POST `/chat`
Send a message to the chatbot.

**Request:**
```json
{
    "message": "Hello, how are you?",
    "user_id": "user123"
}
```

**Response:**
```json
{
    "response": "Hello! I'm doing great, thank you for asking. How can I help you today?"
}
```

### GET `/stats`
Get chatbot usage statistics.

**Response:**
```json
{
    "total_users": 5,
    "total_interactions": 25,
    "active_users": 3
}
```

## 🛠️ Technical Details

### Dependencies
- **Flask**: Web framework
- **OpenAI**: API client for GPT models
- **Transformers**: Hugging Face models for local AI
- **Scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Python-dotenv**: Environment variable management

### Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │───▶│   Flask Server  │───▶│   OpenAI API    │
│   (HTML/JS)     │    │   (Python)      │    │   (GPT-3.5)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Local Models   │
                       │ (DialoGPT/ML)  │
                       └─────────────────┘
```

### ML Pipeline
1. **Input Processing**: Text preprocessing and validation
2. **Intent Classification**: TF-IDF + Naive Bayes classifier
3. **Sentiment Analysis**: RoBERTa-based sentiment model
4. **Response Generation**: OpenAI API or local model
5. **Personalization**: User history-based response adaptation
6. **Output**: Enhanced response with context

## 🎯 Use Cases

### Personal Assistant
- Answer questions and provide information
- Engage in casual conversation
- Remember user preferences

### Customer Support
- Intent recognition for common queries
- Sentiment-aware responses
- Personalized assistance

### Educational Tool
- Fact-based responses
- Interactive learning
- Adaptive explanations

## 🔒 Security & Privacy

- **No Data Persistence**: User data is stored in memory only
- **API Key Security**: Environment variables for sensitive data
- **Input Validation**: Sanitized user inputs
- **Error Handling**: Graceful fallbacks for all failures

## 🚀 Performance Optimizations

- **Response Caching**: Common responses cached for speed
- **Model Optimization**: CPU-optimized local models
- **Async Processing**: Non-blocking response generation
- **Memory Management**: Limited cache sizes

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Check your API key in `.env` file
   - Verify internet connection
   - The chatbot will automatically fall back to local models

2. **Model Loading Issues**
   - Ensure all dependencies are installed
   - Check available disk space for model downloads
   - Verify internet connection for initial model download

3. **Performance Issues**
   - First run may be slow due to model downloads
   - Subsequent runs will be faster
   - Consider using smaller models for limited resources

### Debug Mode
Enable debug mode by setting `DEBUG=true` in your `.env` file for detailed logging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- Hugging Face for transformer models
- Flask community for the web framework
- Scikit-learn team for ML utilities

---

**Happy Chatting! 🤖✨** 
