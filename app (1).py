from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import google.generativeai as genai

app = Flask(__name__, template_folder='templates')


# Load the processors and models for different types of images
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)

# Gemini API Configuration

genai.configure(api_key="AIzaSyD-hL6xW6-puY6i7jqqfs7ePyZwuU2KbYY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    
    # Process the image
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

    # Call Gemini API to process the caption
    response = genai.generate_content(model="gemini-2.0-flash", contents=[{"text": caption}])

    return jsonify({'caption': caption, 'gemini_response': response.text})

if __name__ == '__main__':
    app.run(debug=True)

