from flask import Flask, request, jsonify, render_template_string
import base64
import requests
import json
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load API keys from environment variables
VISION_API_KEY = os.getenv('VISION_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Verify that API keys are loaded
if not all([VISION_API_KEY, GEMINI_API_KEY]):
    logging.error("One or more API keys are missing from the .env file.")
    raise ValueError("One or more API keys are missing from the .env file.")

# Global variable to store the latest processed text
global_context = {
    'sanskrit_text': '',
    'english_text': '',
    'interpretation': ''
}

# Step 1: Extract text using Google Cloud Vision API
def extract_text_from_image(image_content, language_hint="sa"):
    """Extract Sanskrit text from an image using Google Cloud Vision API."""
    try:
        # Convert image to base64
        encoded_image = base64.b64encode(image_content).decode('utf-8')
        
        # Prepare request to Google Cloud Vision API
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": encoded_image
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ],
                    "imageContext": {
                        "languageHints": [language_hint]  # 'sa' for Sanskrit
                    }
                }
            ]
        }
        
        # Make the API request
        vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
        response = requests.post(
            vision_api_url,
            data=json.dumps(request_data),
            headers={'Content-Type': 'application/json'}
        )
        
        # Process the response
        result = response.json()
        
        # Check for errors
        if 'error' in result:
            raise ValueError(f"Vision API error: {result['error']['message']}")
        
        # Extract text from response
        if (result.get('responses') and 
            result['responses'][0].get('textAnnotations') and 
            len(result['responses'][0]['textAnnotations']) > 0):
            extracted_text = result['responses'][0]['textAnnotations'][0]['description']
        else:
            extracted_text = ""
        
        logging.info(f"Extracted text: {extracted_text}")
        return extracted_text.strip()
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return None

# Step 2: Transliterate and Translate to English using Gemini API
def transliterate_sanskrit(sanskrit_text):
    """Transliterate Sanskrit text from Devanagari to IAST with a sanity check."""
    try:
        transliterated_text = transliterate(sanskrit_text, sanscript.DEVANAGARI, sanscript.IAST)
        transliterated_text = ''.join(c for c in transliterated_text if c.isalpha() or c in ' -')
        return transliterated_text
    except Exception as e:
        logging.error(f"Error in transliterate_sanskrit: {str(e)}")
        raise

def translate_to_english(text, source_language="sa", target_language="en"):
    """Translate the text to English using Gemini API."""
    try:
        # Prepare the prompt for Gemini API
        prompt = (
            f"Translate the following {source_language} text to {target_language}:\n"
            f"Text: {text}\n"
            f"Provide only the translated text in {target_language}."
        )
        
        # Prepare the request to Gemini API
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        request_data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        # Make the API request
        response = requests.post(
            gemini_api_url,
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Process the response
        result = response.json()
        
        # Check for errors
        if 'error' in result:
            raise ValueError(f"Gemini API error: {result['error']['message']}")
        
        # Extract the translated text
        if (result.get('candidates') and 
            len(result['candidates']) > 0 and 
            result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts') and 
            len(result['candidates'][0]['content']['parts']) > 0):
            translated_text = result['candidates'][0]['content']['parts'][0]['text']
        else:
            translated_text = ""
        
        return translated_text.strip()
    except Exception as e:
        logging.error(f"Error in translate_to_english: {str(e)}")
        raise

# Step 3: Interpret the English Text using Gemini API
def interpret_text(english_text):
    """Interpret the translated English text using Gemini API."""
    try:
        # Prepare the prompt for Gemini API
        prompt = (
            f"The following is a translated excerpt from an ancient Sanskrit manuscript:\n"
            f"Text: {english_text}\n\n"
            f"Provide a concise summary (2-3 sentences) of the text and a brief contextual analysis "
            f"(1-2 sentences) identifying if it contains spiritual, philosophical, or historical insights typical of Vedic literature."
        )
        
        # Prepare the request to Gemini API
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        request_data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        # Make the API request
        response = requests.post(
            gemini_api_url,
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Process the response
        result = response.json()
        
        # Check for errors
        if 'error' in result:
            raise ValueError(f"Gemini API error: {result['error']['message']}")
        
        # Extract the interpretation
        if (result.get('candidates') and 
            len(result['candidates']) > 0 and 
            result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts') and 
            len(result['candidates'][0]['content']['parts']) > 0):
            interpretation = result['candidates'][0]['content']['parts'][0]['text']
        else:
            interpretation = "Interpretation failed."
        
        return interpretation.strip()
    except Exception as e:
        logging.error(f"Error in interpret_text: {str(e)}")
        raise

# Step 4: Chatbot using Gemini API
def query_gemini_api(user_query):
    """Send a query to the Gemini API with the stored context."""
    try:
        # Prepare the context for the Gemini API
        context = (
            f"The following is an excerpt from an ancient Sanskrit manuscript:\n"
            f"Sanskrit Text: {global_context['sanskrit_text']}\n"
            f"English Translation: {global_context['english_text']}\n"
            f"Interpretation: {global_context['interpretation']}\n\n"
            f"User Query: {user_query}"
        )
        
        # Prepare the request to Gemini API
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        request_data = {
            "contents": [
                {
                    "parts": [
                        {"text": context}
                    ]
                }
            ]
        }
        
        # Make the API request
        response = requests.post(
            gemini_api_url,
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Process the response
        result = response.json()
        
        # Check for errors
        if 'error' in result:
            raise ValueError(f"Gemini API error: {result['error']['message']}")
        
        # Extract the response text
        if (result.get('candidates') and 
            len(result['candidates']) > 0 and 
            result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts') and 
            len(result['candidates'][0]['content']['parts']) > 0):
            response_text = result['candidates'][0]['content']['parts'][0]['text']
        else:
            response_text = "Sorry, I couldn't generate a response."
        
        return response_text
    except Exception as e:
        logging.error(f"Error in query_gemini_api: {str(e)}")
        return f"Error: {str(e)}"

# Flask Routes
@app.route('/')
def index():
    """Render the main page with the upload form and sections."""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VedaLipi - Ancient Script Translator</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Lora:wght@400;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Poppins', sans-serif;
            }
            body {
                background-color: #fff5e6;
                color: #3c2f2f;
            }
            .navbar {
                background: linear-gradient(to bottom, #ff9933, #e68a00);
                padding: 20px 50px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                position: fixed;
                width: 100%;
                top: 0;
                z-index: 1000;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }
            .navbar .logo {
                color: #ffd700;
                font-family: 'Lora', serif;
                font-size: 28px;
                font-weight: 700;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
            }
            .navbar ul {
                list-style: none;
                display: flex;
                gap: 30px;
            }
            .navbar ul li a {
                color: #ffffff;
                text-decoration: none;
                font-size: 16px;
                font-weight: 600;
                transition: color 0.3s ease;
            }
            .navbar ul li a:hover {
                color: #ffd700;
            }
            .section {
                padding: 120px 50px;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            }
            #home {
                position: relative;
                color: #ffffff;
                text-align: center;
                overflow: hidden;
            }
            #home video {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                z-index: 0;
            }
            #home::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.3);
                z-index: 1;
            }
            #home .upload-box {
                position: relative;
                z-index: 2;
                background: rgba(255, 245, 230, 0.9);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
                max-width: 500px;
                margin: 0 auto;
                border: 2px solid #ffd700;
            }
            #home .upload-box h2 {
                font-family: 'Lora', serif;
                color: #3c2f2f;
                margin-bottom: 20px;
                font-size: 28px;
            }
            #home .upload-box p {
                color: #3c2f2f;
                margin-bottom: 20px;
            }
            #home .upload-box input[type="file"] {
                display: none;
            }
            #home .upload-box label {
                background: linear-gradient(to right, #ffd700, #ffcc00);
                color: #3c2f2f;
                padding: 12px 25px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            #home .upload-box label:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }
            #translation {
                background: url('https://mithunonthe.net/wp-content/uploads/2015/02/Hampi-Virupaksha-temple-ancient-ruins-royal-enclosure-architecture-photos-14.jpg') no-repeat center/cover;
                display: flex;
                flex-direction: column;
                gap: 40px;
                position: relative;
            }
            #translation::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 245, 230, 0.8);
                z-index: 1;
            }
            .translation-step {
                position: relative;
                z-index: 2;
                background: rgba(255, 153, 51, 0.1);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
                max-width: 800px;
                width: 100%;
                text-align: center;
                border-left: 5px solid #ffd700;
            }
            .translation-step h3 {
                font-family: 'Lora', serif;
                color: #3c2f2f;
                margin-bottom: 15px;
                font-size: 24px;
            }
            .translation-step p {
                color: #4a3c31;
                font-size: 16px;
            }
            #chatbot {
                background: url('https://images.unsplash.com/photo-1545486336-5a3b6f7a1c7b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80') no-repeat center/cover;
                color: #ffffff;
                text-align: center;
                position: relative;
            }
            #chatbot::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.6);
                z-index: 1;
            }
            .chatbot-container {
                position: relative;
                z-index: 2;
                max-width: 600px;
                width: 100%;
                background: rgba(255, 245, 230, 0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
                border: 2px solid #ff9933;
            }
            .chatbot-container h2 {
                font-family: 'Lora', serif;
                color: #3c2f2f;
                margin-bottom: 20px;
                font-size: 26px;
            }
            .chatbot-messages {
                height: 300px;
                overflow-y: auto;
                background: #fff5e6;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid #ffd700;
            }
            .chatbot-messages p {
                color: #3c2f2f;
                margin-bottom: 10px;
            }
            .chatbot-messages .user-message {
                text-align: right;
                font-weight: 600;
            }
            .chatbot-messages .bot-message {
                text-align: left;
            }
            .chatbot-input {
                display: flex;
                gap: 10px;
            }
            .chatbot-input input {
                flex: 1;
                padding: 12px;
                border: 1px solid #ff9933;
                border-radius: 8px;
                font-size: 16px;
                background: #fff;
            }
            .chatbot-input button {
                background: linear-gradient(to right, #ffd700, #ffcc00);
                color: #3c2f2f;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .chatbot-input button:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }
            #recommendation {
                background: url('https://images.unsplash.com/photo-1563984689740-f37ca8bc3c72?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80') no-repeat center/cover;
                display: flex;
                gap: 40px;
                padding: 50px;
                position: relative;
            }
            #recommendation::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 245, 230, 0.8);
                z-index: 1;
            }
            .recommendation-sidebar {
                position: relative;
                z-index: 2;
                flex: 1;
                background: rgba(255, 153, 51, 0.1);
                padding: 20px;
                border-radius: 15px;
                max-width: 300px;
                border: 2px solid #ffd700;
            }
            .recommendation-sidebar h3 {
                font-family: 'Lora', serif;
                color: #3c2f2f;
                margin-bottom: 20px;
                font-size: 22px;
            }
            .recommendation-sidebar ul {
                list-style: none;
            }
            .recommendation-sidebar ul li {
                margin-bottom: 15px;
            }
            .recommendation-sidebar ul li a {
                color: #3c2f2f;
                text-decoration: none;
                font-weight: 600;
                font-size: 16px;
                transition: color 0.3s ease, transform 0.3s ease;
            }
            .recommendation-sidebar ul li a:hover {
                color: #ffd700;
                transform: translateX(5px);
            }
            .recommendation-content {
                position: relative;
                z-index: 2;
                flex: 2;
                background: rgba(255, 245, 230, 0.95);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
                border: 2px solid #ff9933;
            }
            .recommendation-content h2 {
                font-family: 'Lora', serif;
                color: #3c2f2f;
                margin-bottom: 20px;
                font-size: 26px;
            }
            .recommendation-content p {
                color: #4a3c31;
                font-size: 16px;
            }
            @media (max-width: 768px) {
                .navbar {
                    flex-direction: column;
                    gap: 15px;
                    padding: 15px;
                }
                .navbar ul {
                    flex-direction: column;
                    gap: 15px;
                    text-align: center;
                }
                .section {
                    padding: 100px 20px;
                }
                #recommendation {
                    flex-direction: column;
                }
                .recommendation-sidebar, .recommendation-content {
                    max-width: 100%;
                }
                .upload-box, .translation-step, .chatbot-container {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="logo">VedaLipi</div>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#translation">Translation</a></li>
                <li><a href="#chatbot">Chatbot</a></li>
                <li><a href="#recommendation">Recommendations</a></li>
            </ul>
        </nav>
        <section id="home" class="section">
            <video autoplay muted loop playsinline>
                <source src="/static/background-video.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="upload-box">
                <h2>Upload Ancient Script</h2>
                <p>Begin your journey by uploading an ancient script dataset</p>
                <form id="upload-form" enctype="multipart/form-data">
                    <input type="file" id="file-upload" name="file" accept=".pdf,.jpg,.png">
                    <label for="file-upload">Choose File</label>
                </form>
            </div>
        </section>
        <section id="translation" class="section">
            <div class="translation-step">
                <h3>Step 1: Raw Data to Text</h3>
                <p id="sanskrit-text">Convert the uploaded ancient script into readable text format.</p>
            </div>
            <div class="translation-step">
                <h3>Step 2: Translation to English</h3>
                <p id="english-text">Translate the extracted text into English for better understanding.</p>
            </div>
            <div class="translation-step">
                <h3>Step 3: Interpretation</h3>
                <p id="interpretation-text">Interpret the translated text to uncover its meaning and context.</p>
            </div>
        </section>
        <section id="chatbot" class="section">
            <div class="chatbot-container">
                <h2>Ask Our Ancient Script Expert</h2>
                <div class="chatbot-messages" id="chatbot-messages">
                    <p class="bot-message">Welcome! Ask me anything about the translated script.</p>
                </div>
                <div class="chatbot-input">
                    <input type="text" id="chatbot-input" placeholder="Type your question...">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </section>
        <section id="recommendation" class="section">
            <div class="recommendation-sidebar">
                <h3>Recommended Scripts</h3>
                <ul>
                    <li><a href="#">Vedic Manuscript</a></li>
                    <li><a href="#">Tamil Grantha</a></li>
                    <li><a href="#">Sanskrit Devanagari</a></li>
                </ul>
            </div>
            <div class="recommendation-content">
                <h2>Script Interpretation</h2>
                <p>Select a script from the left to view its detailed interpretation here.</p>
            </div>
        </section>
        <script>
            // Handle image upload and processing
            document.getElementById('file-upload').addEventListener('change', function() {
                const formData = new FormData();
                formData.append('file', this.files[0]);
                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('sanskrit-text').innerText = data.sanskrit_text || 'No text extracted.';
                    document.getElementById('english-text').innerText = data.english_text || 'Translation failed.';
                    document.getElementById('interpretation-text').innerText = data.interpretation || 'Interpretation failed.';
                    window.location.hash = 'translation';
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while processing the image.');
                });
            });

            // Handle chatbot messaging
            function sendMessage() {
                const input = document.getElementById('chatbot-input');
                const message = input.value.trim();
                if (!message) return;

                // Display user message
                const messagesDiv = document.getElementById('chatbot-messages');
                const userMessage = document.createElement('p');
                userMessage.className = 'user-message';
                userMessage.innerText = message;
                messagesDiv.appendChild(userMessage);

                // Clear input
                input.value = '';

                // Send message to server
                fetch('/chatbot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    // Display bot response
                    const botMessage = document.createElement('p');
                    botMessage.className = 'bot-message';
                    botMessage.innerText = data.response || 'Sorry, I couldn’t respond.';
                    messagesDiv.appendChild(botMessage);

                    // Scroll to the bottom
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                })
                .catch(error => {
                    console.error('Error:', error);
                    const botMessage = document.createElement('p');
                    botMessage.className = 'bot-message';
                    botMessage.innerText = 'Error: Unable to get a response.';
                    messagesDiv.appendChild(botMessage);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                });
            }

            // Allow pressing Enter to send message
            document.getElementById('chatbot-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/process', methods=['POST'])
def process_image():
    """Process the uploaded image through the ML pipeline."""
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the image file directly into memory
        image_content = file.read()
        
        # Step 1: Extract text using Google Cloud Vision API
        sanskrit_text = extract_text_from_image(image_content)
        if sanskrit_text is None or sanskrit_text == "":
            raise ValueError("Text extraction failed")
        
        # Step 2: Transliterate and translate using Gemini API
        transliterated_text = transliterate_sanskrit(sanskrit_text)
        english_text = translate_to_english(sanskrit_text)  # Now uses Gemini API
        if not english_text:
            raise ValueError("Translation failed")
        
        # Step 3: Interpret the translated text using Gemini API
        interpretation = interpret_text(english_text)
        
        # Store the results in global context for the chatbot
        global_context['sanskrit_text'] = sanskrit_text
        global_context['english_text'] = english_text
        global_context['interpretation'] = interpretation
        
        return jsonify({
            'sanskrit_text': sanskrit_text,
            'english_text': english_text,
            'interpretation': interpretation
        })
    except Exception as e:
        logging.error(f"Error in process_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Handle chatbot queries using the Gemini API."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_query = data['message']
        response = query_gemini_api(user_query)
        
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error in chatbot: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6005)