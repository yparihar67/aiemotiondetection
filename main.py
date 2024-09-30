import base64
import json
import os
from urllib import response

import pandas as pd
import requests
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_sqlalchemy import SQLAlchemy
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename

# import numpy as np

app = Flask(__name__)
API_KEY = 'LxJ3r6ZYEoH2iAQwPNntpe51neaf0lfE'  # API key
AGENT_ID = 'ag:373a09c1:20240915:emotion-prediction:bafdd063'
AGENT_ENDPOINT = f'https://api.mistral.ai/v1/agents/{AGENT_ID}/execute'

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///emotionapp.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = "secret@1234"

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Class
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vigilant-art-424011-d7-a8f29f0e2c16.json"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Map sentiments to integer values
sentiment_mapping = {
    0: 'happy',
    1: 'sad',
    2: 'angry',
    3: 'fear',
    4: 'neutral',
    5: 'surprise'
}

def predict_text_sentiment_analysis_sample(
    project: str,
    endpoint_id: str,
    content: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instance = aiplatform.gapic.schema.predict.instance.TextSentimentPredictionInstance(
        content=content,
    ).to_value()
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # Extract the first prediction result
    prediction = response.predictions[0]
    sentiment = prediction['sentiment']
    return sentiment

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)

    # Extract predictions
    predictions = response.predictions
    print(predictions)

    # Loop through predictions and print only displayNames
    for prediction in predictions:
        display_names = prediction.get('displayNames', [])
        return display_names[0] if display_names else None
    
with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# SIGNUP Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        cnfPassword = request.form.get('cnfPassword')

        if password != cnfPassword:
            flash('Passwords do not match!', 'danger')
            return redirect('/signup')

        new_user = User(username=username, email=email, password=password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Sign-Up Successful! Please log in.', 'success')
            return redirect('/login')
        except IntegrityError:
            db.session.rollback()
            flash('Username or Email already exists!', 'danger')
            return redirect('/signup')

    return render_template("signup.html")

# LOGIN Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and password == user.password:
            login_user(user)
            flash('Login Successful!', 'success')
            return redirect('/')
        else:
            flash('Invalid email or password!', 'danger')
            return redirect('/login')

    return render_template('login.html')


# LOGOUT Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_prediction')
@login_required
def image_prediction():
    return render_template('image_prediction.html')

@app.route('/text_prediction')
@login_required
def text_prediction():
    return render_template('text_prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/support')
def support():
    return render_template('support.html')


@app.route('/predict_emotion', methods=['POST'])
@login_required
def predict_emotion():
    user_text = request.form.get('user_text')
    sentiment = predict_text_sentiment_analysis_sample(
        project="448446115791",
        endpoint_id="2575250845809508352",
        location="us-central1",
        content=user_text
    )

    # Map the numerical sentiment to text labels
    sentiment_label = sentiment_mapping.get(sentiment, 'neutral')

    # Render the appropriate template based on the sentiment label
    return render_template(f'{sentiment_label}.html')
    
@app.route('/image_prediction', methods=['GET', 'POST'])
@login_required
def image_predictions():
    if request.method == 'POST':
        # Get the image from the POST request
        image_file = request.files.get('image')

        if image_file:
            # Save the file to a temporary location
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(file_path)

            # Call the prediction function
            prediction_label = predict_image_classification_sample(
                project="448446115791",
                endpoint_id="7186936864236896256",
                location="us-central1",
                filename=file_path
            )
            
            print(f"Prediction label: {prediction_label}")  # Debugging line

            if prediction_label:
                # Render the template corresponding to the prediction label
                return render_template(f'{prediction_label}.html')
            else:
                # Return a meaningful message if no prediction could be made
                return "Prediction could not be made", 500
        else:
            # Return a response if no file was provided
            return "No file uploaded", 400
      

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = data['message'].lower()  # Convert to lowercase for case-insensitive comparison

    # Detect if the message is related to web app or predictions
    web_app_keywords = [
        "web app", "website", "app", "application", "portal", "platform", 
        "prediction", "don't know what i'm feeling", "don't know what we are feeling"
    ]
    is_web_app_related = any(keyword in user_message for keyword in web_app_keywords)

    # System prompt based on user message context
    if is_web_app_related:
        system_prompt = (
            "You are a virtual assistant guiding users to utilize our AI prediction tools. "
            "Encourage them to describe their feelings and use the text prediction tool to predict emotions with url http://127.0.0.1:8080/text_prediction  based on their input and output will be the user emotion. "
            "Alternatively, they can upload an image of their face to use the image prediction tool for emotion detection with url http://127.0.0.1:8080/image_prediction.and output will be the user emotion. "
            "Assist them with navigating these tools and encourage them to use our web app. And also mainly use the data from which you were finetuned and instructed"
        )
    else:
        system_prompt = (
            "You are a mental health coach. Offer supportive advice to the user based on their feelings."
            "Provide compassionate and empathetic responses to their concerns.  And also mainly use the data from which you were finetuned and instructed"
        )

    # Create the payload with the dynamic system prompt
    payload = {
        'model': 'ft:open-mistral-7b:373a09c1:20240915:e114ceec',
        'messages': [
                      {'role': 'user', 'content': user_message},
                      {'role': 'system', 'content': system_prompt},
        ],
        'temperature': 0.4,
        'max_tokens': 256,
        'stream': False,
    }

    try:
        # Make the request to the Mistral API
        response = requests.post('https://api.mistral.ai/v1/chat/completions', json=payload, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
        })

        if response.status_code != 200:
            print(f"Error from Mistral API: {response.status_code} - {response.text}")
            return jsonify({'error': f"Mistral API returned an error: {response.status_code}"}), 500

        response_data = response.json()

        assistant_response = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        if not assistant_response:
            return jsonify({'error': 'No response content received from Mistral'}), 500

        return jsonify({'response': assistant_response})

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
