# Gemini API Integration Guide

This guide shows you how to integrate Google's free Gemini API into the Disaster Prediction Bot for AI-powered disaster analysis and recommendations.

## 1. Get Your Free Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Get API Key"** or **"Create API Key"**
4. Copy your API key

> **Note**: The free tier includes generous limits - great for development and testing!

---

## 2. Add API Key to Environment

Add your Gemini API key to the `.env` file:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 3. Install Gemini SDK

Add the Google Generative AI package to your virtual environment:

```bash
source .venv/bin/activate
pip install google-generativeai
```

Update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## 4. Example Integration Code

Here's how to integrate Gemini into your disaster prediction bot:

### Basic Setup

```python
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash-exp')
```

### Use Case 1: Disaster Risk Analysis

Add this function to `disaster_mission_control.py`:

```python
def get_ai_disaster_analysis(location, current_risks, weather_data):
    """Get AI-powered disaster analysis using Gemini"""
    
    # Prepare context
    prompt = f'''
    You are a disaster risk analyst. Analyze the following situation:
    
    Location: {location}
    Current Weather: Temperature {weather_data.get('temp')}°C, 
                     Humidity {weather_data.get('humidity')}%
    
    Risk Assessment:
    {chr(10).join([f"- {disaster}: {risk}%" for disaster, risk in current_risks.items()])}
    
    Provide:
    1. Top 3 immediate concerns
    2. Specific safety recommendations
    3. Timeline of when to take action
    
    Keep response under 200 words, bullet points preferred.
    '''
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"
```

### Use Case 2: Emergency Response Recommendations

```python
def get_emergency_recommendations(disaster_type, risk_level):
    """Get AI-generated emergency recommendations"""
    
    prompt = f'''
    Generate emergency preparedness steps for:
    Disaster: {disaster_type}
    Risk Level: {risk_level}%
    
    Provide 5 specific, actionable steps in order of priority.
    Format as numbered list.
    '''
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate recommendations: {str(e)}"
```

### Use Case 3: Interactive Chat Assistant

Add this to your Streamlit UI:

```python
# In your main() function, add a chat interface
st.subheader("🤖 AI Disaster Assistant")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input("Ask about disaster preparedness...")

if user_question:
    # Build context with current data
    context = f'''
    Current location data:
    - Risks: {current_risks}
    - Weather: {weather_data}
    
    User question: {user_question}
    '''
    
    # Get AI response
    chat = model.start_chat(history=st.session_state.chat_history)
    response = chat.send_message(context)
    
    # Update chat history
    st.session_state.chat_history.append({
        "role": "user",
        "parts": [user_question]
    })
    st.session_state.chat_history.append({
        "role": "model",
        "parts": [response.text]
    })
    
    # Display conversation
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["parts"][0])
```

---

## 5. Example: Enhanced Satellite Risk Scan

Enhance your existing risk scan with AI insights:

```python
# After generating current_risks
if os.getenv('GEMINI_API_KEY'):
    with st.expander("🤖 AI Analysis", expanded=False):
        with st.spinner("Analyzing risks with AI..."):
            ai_analysis = get_ai_disaster_analysis(
                location=f"{st.session_state.form_city}",
                current_risks=current_risks,
                weather_data=current_weather
            )
            st.markdown(ai_analysis)
```

---

## 6. Best Practices

### Rate Limiting
```python
import time
from functools import wraps

def rate_limit(max_calls_per_minute=15):
    """Decorator to rate limit API calls"""
    min_interval = 60.0 / max_calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(max_calls_per_minute=15)
def call_gemini_api(prompt):
    """Rate-limited Gemini API call"""
    response = model.generate_content(prompt)
    return response.text
```

### Error Handling
```python
def safe_ai_call(prompt, fallback_message="AI unavailable"):
    """Safely call Gemini with error handling"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"AI feature unavailable: {str(e)}")
        return fallback_message
```

### Caching Responses
```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_ai_analysis(location, risk_summary):
    """Cached AI analysis to reduce API calls"""
    prompt = f"Analyze disaster risks for {location}: {risk_summary}"
    return model.generate_content(prompt).text
```

---

## 7. Free Tier Limits

**Gemini 2.0 Flash (Free Tier)**:
- ✅ 15 requests per minute
- ✅ 1,500 requests per day
- ✅ 1 million tokens per minute

**Tips to stay within limits**:
- Cache repeated queries
- Batch user questions
- Use rate limiting decorators
- Implement smart context trimming

---

## 8. Quick Integration Checklist

- [ ] Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- [ ] Add `GEMINI_API_KEY` to `.env`
- [ ] Install `google-generativeai` package
- [ ] Import and configure in your code
- [ ] Add AI analysis features to UI
- [ ] Implement error handling
- [ ] Test with sample queries

---

## 9. Example Use Cases for Your Bot

1. **Disaster Explanation**: "Why is my flood risk at 15%?"
2. **Preparation Guide**: "How should I prepare for the drought risk?"
3. **Historical Context**: "What were past disasters in this area?"
4. **Forecast Analysis**: "Explain the 7-day forecast trends"
5. **Emergency Contacts**: "What emergency services should I contact?"

---

## 10. Full Working Example

```python
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Add to your Streamlit app
st.title("🌍 AI-Enhanced Disaster Prediction")

# Your existing risk calculation
current_risks = {
    'Flood': 12, 'Heatwave': 8, 'Cyclone': 5
    # ... other risks
}

# Add AI insights
if st.button("Get AI Recommendations"):
    with st.spinner("Consulting AI..."):
        prompt = f'''
        Based on these disaster risks: {current_risks}
        Provide 3 specific safety actions to take right now.
        '''
        response = model.generate_content(prompt)
        st.success(response.text)
```

---

## Resources

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Python Quickstart](https://ai.google.dev/tutorials/python_quickstart)
- [API Key Management](https://aistudio.google.com/app/apikey)
- [Pricing & Limits](https://ai.google.dev/pricing)

---

**Need Help?** Check the [Gemini API Community Forum](https://discuss.ai.google.dev/)
