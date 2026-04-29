import os
import pickle
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
import cv2
from werkzeug.utils import secure_filename
from processing import preprocess_image, extract_features, is_valid_image

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dietary recommendations mapping
RECOMMENDATIONS = {
    'Vitamin A Deficiency': "Increase intake of carrots, sweet potatoes, spinach, and liver. Consider a Vitamin A supplement.",
    'Vitamin B12 Deficiency': "Consume more meat, fish, milk, cheese, and eggs. Fortified cereals or a B12 supplement are recommended for vegetarians/vegans.",
    'Vitamin C Deficiency': "Eat more citrus fruits (oranges, lemons), strawberries, bell peppers, and broccoli.",
    'Vitamin D Deficiency': "Increase sun exposure safely. Eat fatty fish, egg yolks, and fortified dairy. Consider a Vitamin D supplement.",
    'Iron/Anaemia': "Include red meat, beans, lentils, dark leafy greens, and iron-fortified cereals in your diet. Pair with Vitamin C for better absorption.",
    'Healthy': "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins."
}

def analyze_image_with_gemini(filepath):
    try:
        import PIL.Image
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "ERROR", "Unknown", "API Key missing in production environment. Please wait for Render to finish deploying."
            
        genai.configure(api_key=api_key)
        
        with PIL.Image.open(filepath) as img:
            model = genai.GenerativeModel('gemini-flash-latest')
            prompt = """
        You are an expert clinical AI analyzing this image.
        First, determine if it is a clinical close-up of a human tongue, eye, nail, or skin.
        If it is NOT (e.g. it is a poster, object, car, animal, multiple people), reply strictly with:
        STATUS: INVALID
        REASON: [Brief reason why]
        
        If it IS a valid clinical image, evaluate it for signs of these deficiencies:
        - Vitamin A Deficiency
        - Vitamin B12 Deficiency
        - Vitamin C Deficiency
        - Vitamin D Deficiency
        - Iron/Anaemia
        - Healthy
        
        Reply strictly with:
        STATUS: VALID
        DIAGNOSIS: [Choose EXACTLY ONE from the list above]
        REASON: [Brief medical reason based on the visual evidence]
        """
            response = model.generate_content([prompt, img])
            text = response.text
        
        status = "INVALID"
        diagnosis = "Unknown"
        reason = "Could not parse response"
        
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("STATUS:"):
                status = line.split("STATUS:", 1)[1].strip()
            elif line.startswith("DIAGNOSIS:"):
                diagnosis = line.split("DIAGNOSIS:", 1)[1].strip()
            elif line.startswith("REASON:"):
                reason = line.split("REASON:", 1)[1].strip()
                
        return status, diagnosis, reason
    except Exception as e:
        return "ERROR", "Unknown", str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Use Gemini AI for definitive validation and diagnosis
                status, diagnosis, reason = analyze_image_with_gemini(filepath)
                
                if status == "INVALID":
                    flash(f"Invalid Image: {reason}")
                    os.remove(filepath)
                    return redirect(request.url)
                elif status == "ERROR":
                    flash(f"AI Error: {reason}")
                    os.remove(filepath)
                    return redirect(request.url)

                # Still run background extraction to display the prototype metrics
                resized_img, lab_img, mask = preprocess_image(filepath)
                features = extract_features(resized_img, lab_img, mask)
                
                prediction = diagnosis
                recommendation = RECOMMENDATIONS.get(prediction, reason)
                
                # Cleanup uploaded file
                os.remove(filepath)
                
                return render_template('index.html', 
                                       prediction=prediction, 
                                       recommendation=recommendation,
                                       features_extracted=len(features))
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                return redirect(request.url)
                
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
