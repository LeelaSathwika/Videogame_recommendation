from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress TensorFlow logs (if using deep learning later)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Ensure CSV exists before loading
CSV_PATH = "C:/Users/undim/OneDrive/Desktop/New folder/game_recommendation/gaming.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

# Load dataset
data = pd.read_csv(CSV_PATH)

# Genre Mapping Dictionary
genre_mapping = {"1": "Battle", "2": "Vehicle", "3": "Thinking/Logical", "4": "Casual"}

# Encode categorical features
genre_encoder = LabelEncoder()
data["Encoded_Genre"] = genre_encoder.fit_transform(data["Genre"])

gender_encoder = LabelEncoder()
data["Encoded_Gender"] = gender_encoder.fit_transform(data["Gender Preference"])

game_encoder = LabelEncoder()
data["Encoded_GameName"] = game_encoder.fit_transform(data["Game Name"])
num_games = len(game_encoder.classes_)

# Scale rating
scaler = StandardScaler()
data["Scaled_Rating"] = scaler.fit_transform(data[["Rating"]])

# Preprocess platform column
def clean_platforms(platforms_str):
    if isinstance(platforms_str, str):
        return [p.strip().lower() for p in platforms_str.split(",") if p.strip()]
    return []

data['Platform_List'] = data['Platform'].apply(clean_platforms)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles game recommendation requests from the frontend."""
    data_request = request.json
    print("\n🔹 Received Request Data:", data_request)

    # Extract user inputs
    genre_key = data_request.get("genre")
    gender = data_request.get("gender")
    rating = float(data_request.get("rating"))
    platforms = [p.strip().lower() for p in data_request.get("platforms", "").split(",") if p.strip()]

    print(f"🎮 Genre: {genre_key}, Gender: {gender}, Rating: {rating}, Platforms: {platforms}")

    # Convert genre key to name
    genre = genre_mapping.get(genre_key)
    if genre is None:
        print("❌ Invalid genre selection.")
        return jsonify([])

    print(f"🔄 Mapped Genre: {genre}")

    # Encode user inputs
    try:
        encoded_genre = genre_encoder.transform([genre])[0]
        encoded_gender = gender_encoder.transform([gender])[0]
        scaled_rating = scaler.transform([[rating]])[0][0]
    except Exception as e:
        print(f"❌ Encoding error: {e}")
        return jsonify([])

    print(f"✅ Encoded Genre: {encoded_genre}, Encoded Gender: {encoded_gender}, Scaled Rating: {scaled_rating}")

    # Filter dataset
    filtered_data = data[
        (data['Encoded_Genre'] == encoded_genre) &
        (data['Encoded_Gender'] == encoded_gender) &
        (data['Scaled_Rating'] >= scaled_rating)
    ]

    print("🔍 Filtered Data Before Platform Check:", filtered_data)

    # Platform filtering
    if platforms:
        filtered_data = filtered_data[filtered_data['Platform_List'].apply(lambda x: any(p in x for p in platforms))]

    print("✅ Final Filtered Data:", filtered_data)

    if filtered_data.empty:
        print("❌ No matching games found.")
        return jsonify([])

    # Select up to 10 recommendations
    recommendations = filtered_data.sample(n=min(10, len(filtered_data))).to_dict(orient="records")

    print("🎯 Recommendations Sent:", recommendations)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)