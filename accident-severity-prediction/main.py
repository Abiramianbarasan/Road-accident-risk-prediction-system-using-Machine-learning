import os
import sys
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory, jsonify, session, redirect, url_for, flash
import pandas as pd
import joblib
import numpy as np
import urllib.request
import urllib.parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from twilio.rest import Client
import json
import logging
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime
import csv
from functools import wraps
import hashlib
# Import security features from werkzeug
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import re

from utils.route_utils import RouteUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db' # Using SQLite
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'road accident'  # Change this to a secure secret key

db = SQLAlchemy(app)

# Define User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    mobile_number = db.Column(db.String(15), unique=True, nullable=True)  # Added mobile number field
    role = db.Column(db.String(20), default='user', nullable=False) # 'user' or 'admin'
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"User('{self.email}', '{self.role}')"

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()
    
    # Create a default admin user if none exists
    if not User.query.filter_by(role='admin').first():
        # *** CHANGE 'admin@example.com' and 'your_admin_password_here' ***
        # *** to your desired default admin credentials ***
        default_admin_email = 'admin@example.com' # Default admin email
        default_admin_password = 'Abilavz@25' # Default admin password
        
        hashed_password = generate_password_hash(default_admin_password)
        default_admin = User(email=default_admin_email, password=hashed_password, role='admin')
        db.session.add(default_admin)
        db.session.commit()
        print(f"Default admin user '{default_admin_email}' created.")

# Remove the old USERS dictionary
# USERS = {
#     'admin': {
#         'password': hashlib.sha256('admin123'.encode()).hexdigest(),
#         'role': 'admin'
#     },
#     'user': {
#         'password': hashlib.sha256('user123'.encode()).hexdigest(),
#         'role': 'user'
#     }
# }

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('index')) # Redirect to index or a permission denied page
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        logger.info(f"Login attempt with email: {email}")
        logger.info(f"Password length: {len(password) if password else 0}")

        user = User.query.filter_by(email=email).first()
        
        if user:
            logger.info(f"User found with email: {email}")
            logger.info(f"User role: {user.role}")
            logger.info(f"Stored password hash: {user.password[:20]}...")  # Log first 20 chars of hash
            
            if check_password_hash(user.password, password):
                logger.info(f"Password match for user: {email}")
                session['user_id'] = user.id
                session['username'] = user.email
                session['role'] = user.role
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                logger.warning(f"Password mismatch for user: {email}")
                logger.warning("Password verification failed. Please check if the password was entered correctly.")
                flash('Invalid email or password', 'error')
        else:
            logger.warning(f"User not found with email: {email}")
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        mobile_number = request.form.get('mobile_number')  # Get mobile number from form


        # Mobile number validation (updated regex for E.164 format)
        if mobile_number:
            # Updated basic mobile number validation for E.164 format
            if not re.match(r'^\+[1-9]\d{1,14}$', mobile_number):
                flash('Please enter a valid mobile number in E.164 format (e.g., +919876543210).', 'error')
                return render_template('register.html')

        # Password Policy Check
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('register.html')
        if not re.search(r'[a-z]', password):
            flash('Password must contain at least one lowercase letter.', 'error')
            return render_template('register.html')
        if not re.search(r'[A-Z]', password):
            flash('Password must contain at least one uppercase letter.', 'error')
            return render_template('register.html')
        if not re.search(r'\d', password):
            flash('Password must contain at least one digit.', 'error')
            return render_template('register.html')
        if not re.search(r'[!@#$%^&*()-_+=]', password):
            flash('Password must contain at least one special character (!@#$%^&*()-_+=).', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already exists.', 'error')
            return render_template('register.html')
        
        # Check if mobile number already exists
        if mobile_number:
            existing_mobile = User.query.filter_by(mobile_number=mobile_number).first()
            if existing_mobile:
                flash('Mobile number already registered.', 'error')
                return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password=hashed_password, mobile_number=mobile_number, role='user')
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Define the path for storing user submissions
USER_SUBMISSIONS_FILE = 'user_submissions.csv'

# Twilio credentials (you'll need to sign up for a free account at twilio.com)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = '+17156247714'

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load and preprocess the data
try:
    # Read the CSV file
    print("Loading CSV file...")
    df = pd.read_csv('cleaned_accident_data_preprocessed (4).csv')
    
    # Print the distribution of accident severities
    print("\nAccident Severity Distribution:")
    print(df['Accident_Severity'].value_counts())
    
    # Convert categorical variables to numerical
    le = LabelEncoder()
    categorical_columns = ['Accident_Severity', 'Light_Conditions', 'Road_Surface_Conditions', 
                         'Weather_Conditions', 'Urban_or_Rural_Area']
    
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
        if col == 'Accident_Severity':
            print("\nEncoded Accident Severity Mapping:")
            for i, label in enumerate(le.classes_):
                print(f"{i}: {label}")
    
    # Select features and target (removed Did_Police_Officer_Attend_Scene_of_Accident)
    features = ['Light_Conditions', 
               'Road_Surface_Conditions', 'Speed_limit', 'Weather_Conditions',
               'Latitude', 'Longitude']
    
    if not all(col in df.columns for col in features):
        print("Error: Some required columns not found in the CSV.")
        missing_cols = [col for col in features if col not in df.columns]
        print(f"Missing columns: {missing_cols}")
        sys.exit(1)
        
    X = df[features]
    y = df['Accident_Severity']
    
    # Print feature information
    print("\nFeatures used for training:")
    for feature in features:
        print(f"- {feature}")
    
    # Create and train the model with balanced class weights and adjusted parameters
    print("\nTraining new model with updated feature set...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced',
        criterion='entropy'
    )
    
    # Train the model
    model.fit(X, y)
    
    # Print feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Save the new model
    model_path = 'trained_model_new.sav'
    joblib.dump(model, model_path)
    print(f"\nNew model saved to {model_path}")
    
    # Load the new model for immediate use
    model = joblib.load(model_path)
    print("New model loaded successfully")
    
except Exception as e:
    print(f"Error during model training: {e}")
    print("Falling back to pre-trained model...")
    try:
        model = joblib.load('litemodel.sav')
        print("Successfully loaded pre-trained model")
    except:
        print("Could not load pre-trained model, using fallback model")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced',
            criterion='entropy'
        )
        # Updated fallback data without Did_Police_Officer_Attend
        X_fallback = np.array([
            [5, 4, 70, 5, 0, 0],  # Fatal case
            [6, 2, 60, 2, 0, 0],  # Fatal case
            [4, 3, 50, 3, 0, 0],  # Serious case
            [7, 1, 65, 7, 0, 0],  # Fatal case
            [1, 5, 55, 2, 0, 0],  # Serious case
            [1, 1, 120, 1, 0, 0], # Fatal case
            [4, 1, 40, 1, 0, 0],  # Serious case
            [1, 2, 45, 2, 0, 0],  # Serious case
            [4, 1, 35, 1, 0, 0],  # Slight case
            [1, 2, 40, 2, 0, 0],  # Slight case
            [1, 1, 45, 4, 0, 0],  # Slight case
            [1, 1, 30, 1, 0, 0],  # Slight case
            [1, 1, 20, 1, 0, 0],  # Slight case
            [1, 1, 25, 1, 0, 0],  # Slight case
            [1, 1, 15, 1, 0, 0],  # Slight case
            [1, 1, 10, 1, 0, 0],  # Slight case
            [1, 1, 30, 1, 0, 0]   # Slight case
        ])
        y_fallback = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        model.fit(X_fallback, y_fallback)
        print("Using fallback model due to training error")

def cal(ip):
    print("DEBUG: Raw input received:", ip)
    input = dict(ip)
    try:
        light = float(input['light'][0])
        roadsc = float(input['roadsc'][0])
        speedl = float(input['speedl'][0])
        weather = float(input['weather'][0])
        latitude = float(input['latitude'][0])
        longitude = float(input['longitude'][0])
        data = np.array([light, roadsc, speedl, weather, latitude, longitude])
        print("\nInput data (including lat/lon):", data)
        data = data.reshape(1, -1)
        try:
            result = model.predict(data)
            probabilities = model.predict_proba(data)[0]
            severity = {0: "Fatal", 1: "Serious", 2: "Slight"}
            print("Prediction result:", severity[result[0]])
            print("Prediction probabilities:", dict(zip(['Fatal', 'Serious', 'Slight'], probabilities)))
            return severity[result[0]]
        except Exception as e:
            print(f"Prediction error: {e}")
            return f"Error in prediction: {str(e)}"
    except Exception as e:
        print(f"Input processing error: {e}")
        return f"Error processing input: {str(e)}"

def store_user_submission(form_data, prediction_result):
    try:
        # Define mappings for categorical features to store as strings
        weather_map = {
            "1": "Fine no high winds",
            "2": "Raining no high winds",
            "3": "Snowing no high winds",
            "4": "Fine + high winds",
            "5": "Raining + high winds",
            "6": "Snowing + high winds",
            "7": "Fog or mist",
            '': 'Unknown' # Handle potential missing value
        }
        light_map = {
            "1": "Daylight",
            "4": "Dark - lights lit",
            "5": "Dark - lights unlit",
            "6": "Dark - no lighting",
             '': 'Unknown' # Handle potential missing value
        }
        roadsc_map = {
            "1": "Dry",
            "2": "Wet",
            "3": "Snow",
            "4": "Frost",
            "5": "Flood",
            "7": "Mud",
             '': 'Unknown' # Handle potential missing value
        }
        day_map = {
            "1": "Sunday",
            "2": "Monday",
            "3": "Tuesday",
            "4": "Wednesday",
            "5": "Thursday",
            "6": "Friday",
            "7": "Saturday",
             '': 'Unknown' # Handle potential missing value
        }
        vehicle_map = {
            "1": "Pedal cycle",
            "2": "Motorcycle 50cc and under",
            "3": "Motorcycle 125cc and under",
            "4": "Motorcycle over 125cc and up to 500cc",
            "5": "Motorcycle over 500cc",
            "8": "Taxi/Private hire car",
            "9": "Car",
            "10": "Minibus (8 - 16 passenger seats)",
            "11": "Bus or coach (17 or more pass seats)",
            "18": "Tram",
            "20": "Truck(Goods)",
            "23": "Electric motorcycle",
             '': 'Unknown' # Handle potential missing value
        }
        gender_map = {
            "1": "Male",
            "2": "Female",
            "3": "Unknown", # Assuming '3' is the unknown value from the form
             '': 'Unknown' # Handle potential missing value
        }

        # Prepare the data to store
        submission_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # Changed key to match HTML header
            'Age of Driver': form_data.get('age_of_driver', ''), # Should capture the age as entered
            'Vehicle Type': vehicle_map.get(form_data.get('vehicle_type', ''), form_data.get('vehicle_type', '')), # Map to string
            'Age of Vehicle': form_data.get('age_of_vehicle', ''), # Changed key to match HTML header
            'Engine Capacity (CC)': form_data.get('engine_cc', ''), # Changed key to match HTML header
            'Day of Week': day_map.get(form_data.get('day', ''), form_data.get('day', '')), # Map to string
            'Weather': weather_map.get(form_data.get('weather', ''), form_data.get('weather', '')), # Changed key to match HTML header
            'Light Conditions': light_map.get(form_data.get('light', ''), form_data.get('light', '')), # Changed key to match HTML header
            'Road Surface': roadsc_map.get(form_data.get('roadsc', ''), form_data.get('roadsc', '')), # Changed key to match HTML header
            'Gender': gender_map.get(form_data.get('gender', ''), form_data.get('gender', '')), # Map to string
            'Speed Limit': form_data.get('speedl', ''), # Changed key to match HTML header
            'Latitude': form_data.get('latitude', ''),
            'Longitude': form_data.get('longitude', ''),
            'Predicted Severity': prediction_result if prediction_result not in ["Error in prediction: ", "Error processing input: "] else "Prediction Error" # Store error message if prediction failed
        }

        # Ensure the keys are in the desired order for the CSV header
        ordered_keys = [
            'Timestamp', 'Age of Driver', 'Vehicle Type', 'Age of Vehicle',
            'Engine Capacity (CC)', 'Day of Week', 'Weather', 'Light Conditions',
            'Road Surface', 'Gender', 'Speed Limit', 'Latitude', 'Longitude',
            'Predicted Severity'
        ]

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(USER_SUBMISSIONS_FILE)
        
        # Write to CSV
        with open(USER_SUBMISSIONS_FILE, 'a', newline='') as f:
            # Use the ordered_keys for the fieldnames
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            if not file_exists:
                writer.writeheader()
            writer.writerow(submission_data)
            
        logger.info(f"Successfully stored user submission with prediction: {prediction_result}")
        return True
    except Exception as e:
        logger.error(f"Error storing user submission: {str(e)}")
        return False

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        try:
            result = cal(request.form)
            store_user_submission(request.form, result)
            return result
        except Exception as e:
            logger.error(f"Error in index route: {str(e)}")
            return f"Error: {str(e)}"
    return render_template('index.html')

@app.route('/submissions', methods=['GET'])
@admin_required
def view_submissions():
    try:
        if not os.path.exists(USER_SUBMISSIONS_FILE):
            return "No submissions found", 404
        
        df = pd.read_csv(USER_SUBMISSIONS_FILE)
        print("Submissions loaded:", df.shape)
        print(df.head())
        return render_template('submissions.html', submissions=df.to_dict('records'))
    except Exception as e:
        logger.error(f"Error viewing submissions: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/visual/', methods=['GET'], strict_slashes=False)
def visual():
    return render_template('visual.html')

@app.route('/leaflet-heatmap', methods=['GET'])
def leaflet_heatmap():
    return render_template('leaflet_heatmap.html')

@app.route('/sms/', methods=['POST'])
def sms():
    try:
        # Get prediction result
        res = cal(request.form)

        # Always use the default Twilio registered number for sending SMS
        phone_number = "+918667334079"  # Default number to send SMS to

        # Extract data from form
        latitude = request.form.get('latitude', 'N/A')
        longitude = request.form.get('longitude', 'N/A')
        weather = weather_labels.get(request.form.get('weather', ''), request.form.get('weather', 'Unknown'))
        light = light_labels.get(request.form.get('light', ''), request.form.get('light', 'Unknown'))
        roadsc = road_labels.get(request.form.get('roadsc', ''), request.form.get('roadsc', 'Unknown'))

        # Custom message based on severity
        if res == "Fatal":
            alert_msg = "FATAL ACCIDENT! Immediate emergency response required."
        elif res == "Serious":
            alert_msg = "Serious accident reported. Please proceed with caution."
        else:
            alert_msg = "Minor accident detected. Be alert on the road."

        # Final SMS content
        message_body = (
            f"{alert_msg}\n"
            f"Severity: {res}\n"
            f"Location: https://www.google.com/maps?q={latitude},{longitude}\n"
            f"Weather: {weather}\n"
            f"Light Condition: {light}\n"
            f"Road Condition: {roadsc}"
        )

        # Send SMS
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )

        logger.info(f"Twilio SMS SID: {message.sid}")
        return f"SMS Alert sent successfully to {phone_number}"

    except Exception as e:
        logger.error(f"Error in SMS route: {str(e)}")
        return f"Error sending SMS: {str(e)}"

@app.route('/webhook/sms', methods=['POST'])
def receive_sms():
    try:
        # Get the message details from Twilio
        from_number = request.values.get('From', None)
        message_body = request.values.get('Body', None)
        
        # Log the received message
        logger.info(f"Received SMS from {from_number}: {message_body}")
        
        # Process the message (you can add your logic here)
        # For example, you might want to:
        # 1. Parse the message for location coordinates
        # 2. Update the prediction model
        # 3. Send a response back
        
        # Send a response back to the sender
        response = client.messages.create(
            body="Thank you for your message. We have received your report.",
            from_=TWILIO_PHONE_NUMBER,
            to=from_number
        )
        
        return "Message received", 200
        
    except Exception as e:
        logger.error(f"Error processing incoming SMS: {str(e)}")
        return "Error processing message", 500

@app.route('/get-api-key')
def get_api_key():
    return jsonify({'api_key': 'AIzaSyBPDqK_MEL2VCprmoR3a-SJ7lVaGrcRqps'})

@app.route('/heatmap-data', methods=['GET'])
def heatmap_data():
    import os
    print("Current working directory:", os.getcwd())
    print("File exists:", os.path.isfile('cleaned_accident_data_with_location.csv'))
    try:
        logger.info("Attempting to load heatmap data...")
        df = pd.read_csv('cleaned_accident_data_with_location.csv')
        logger.info(f"Successfully loaded data with {len(df)} rows")

        # Filter for India (approximate bounding box)
        india_df = df[
            (df['Latitude'] >= 6.0) & (df['Latitude'] <= 38.0) &
            (df['Longitude'] >= 68.0) & (df['Longitude'] <= 98.0)
        ]

        X = india_df[[
            "Did_Police_Officer_Attend_Scene_of_Accident",
            "Age_of_Driver",
            "Vehicle_Type",
            "Age_of_Vehicle",
            "Engine_Capacity_(CC)",
            "Day_of_Week",
            "Weather",
            "Road_Surface",
            "Light",
            "Gender",
            "Speed_limit",
            "Latitude",
            "Longitude"
        ]]

        severity_weights = {
            'Fatal': 1.0,
            'Serious': 0.6,
            'Slight': 0.2
        }

        heatmap_data = []
        for index, row in india_df.iterrows():
            try:
                if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                    weight = severity_weights.get(row['Accident_Severity'], 0.2)
                    heatmap_data.append({
                        'location': {
                            'lat': float(row['Latitude']),
                            'lng': float(row['Longitude'])
                        },
                        'weight': weight
                    })
            except Exception as e:
                logger.error(f"Error processing row {index}: {str(e)}")
                continue

        logger.info(f"Generated {len(heatmap_data)} heatmap points")
        return jsonify(heatmap_data)
    except FileNotFoundError:
        logger.error("Error: Data file not found")
        return jsonify({'error': 'Data file not found'}), 404
    except Exception as e:
        logger.error(f"Error generating heatmap data: {str(e)}")
        return jsonify({'error': str(e)}), 500

weather_labels = {
    "1": "Fine no high winds",
    "2": "Raining no high winds",
    "3": "Snowing no high winds",
    "4": "Fine + high winds",
    "5": "Raining + high winds",
    "6": "Snowing + high winds",
    "7": "Fog or mist"
}
light_labels = {
    "1": "Daylight",
    "4": "Dark - lights lit",
    "5": "Dark - lights unlit",
    "6": "Dark - no lighting"
}
road_labels = {
    "1": "Dry",
    "2": "Wet",
    "3": "Snow",
    "4": "Frost",
    "5": "Flood",
    "7": "Mud"
}

# Load accident data (replace with your actual data loading logic)
try:
    accident_data_path = os.path.join(os.path.dirname(__file__), 'cleaned_accident_data_with_location.csv')
    accident_data = pd.read_csv(accident_data_path).to_dict('records')
except Exception as e:
    accident_data = []
    logger.error(f"Could not load accident data: {e}")

# Placeholder function for getting accident data for a specific area
def get_accident_data_for_area(min_lat, max_lat, min_lon, max_lon):
    # In a real application, you would filter the global accident_data
    # or query a database based on the bounding box.
    # For now, we return the full dataset if loaded, or an empty list.
    if accident_data:
        # Simple filtering (can be optimized) - assuming accident_data has 'latitude' and 'longitude' keys
        filtered_data = [
            acc for acc in accident_data
            if acc.get('Latitude') is not None and acc.get('Longitude') is not None and \
               min_lat <= acc['Latitude'] <= max_lat and \
               min_lon <= acc['Longitude'] <= max_lon
        ]
        return {'accidents': filtered_data}
    return {'accidents': []}

route_utils = RouteUtils()

@app.route('/api/safest-route', methods=['POST'])
def api_safest_route():
    try:
        data = request.get_json()
        start_location = data.get('start')
        end_location = data.get('end')
        
        if not start_location or not end_location:
            return jsonify({'error': 'Start and end locations are required'}), 400
            
        # Get coordinates for start and end locations
        start_coords = route_utils.geocode_address(start_location)
        end_coords = route_utils.geocode_address(end_location)
        
        # Get route alternatives from OpenRouteService
        api_key = os.getenv('OPENROUTE_API_KEY')
        if not api_key:
            return jsonify({'error': 'OpenRoute API key not configured'}), 500
            
        routes = route_utils.get_route_alternatives(start_coords, end_coords, api_key)
        
        if not routes:
            return jsonify({'error': 'No routes found'}), 404
            
        # Get accident data for the area
        accident_data = get_accident_data_for_area(
            min(start_coords[0], end_coords[0]),
            max(start_coords[0], end_coords[0]),
            min(start_coords[1], end_coords[1]),
            max(start_coords[1], end_coords[1])
        )
        
        # Analyze each route and find the safest one
        safest_route = None
        min_score = float('inf')
        
        for route in routes:
            try:
                coordinates = [[pt[1], pt[0]] for pt in route['geometry']['coordinates']]
                
                # Get comprehensive safety analysis
                safety_analysis = route_utils.analyze_route_safety(coordinates, {
                    'Did_Police_Officer_Attend': 1,
                    'Light_Conditions': 1,  # Default to daylight
                    'Road_Surface_Conditions': 1,  # Default to dry
                    'Speed_limit': 30,  # Default speed limit
                    'Weather_Conditions': 1  # Default to fine weather
                })
                
                # Add safety analysis to route object
                route.update({
                    'safety_score': safety_analysis['overall_risk_score'],
                    'risk_level': safety_analysis['risk_level'],
                    'accident_count': safety_analysis['historical_analysis']['accident_count'],
                    'accident_density': safety_analysis['historical_analysis']['accident_density'],
                    'recommendations': safety_analysis['recommendations'],
                    'risk_segments': safety_analysis['risk_segments']
                })
                
                if safety_analysis['overall_risk_score'] < min_score:
                    min_score = safety_analysis['overall_risk_score']
                    safest_route = route
                    
            except Exception as e:
                logger.error(f"Error analyzing route safety: {e}")
                continue

        if not safest_route:
            return jsonify({'error': 'Could not determine safest route'}), 500

        return jsonify({
            'routes': routes,
            'safest_route': safest_route,
            'start_coords': start_coords,
            'end_coords': end_coords
        })

    except Exception as e:
        logger.error(f"Error in /api/safest-route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/safest-route')
@login_required
def safest_route_page():
    return render_template('route.html')

@app.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.all() # Fetch all users from the database
    return render_template('admin.html', users=users)

@app.route('/reset-admin-password', methods=['GET'])
def reset_admin_password():
    try:
        with app.app_context():
            # First, let's check if the database exists and is accessible
            logger.info("Starting admin password reset process...")
            
            # Find and delete the existing admin user(s) if they exist
            existing_admins = User.query.filter_by(email='admin@example.com').all()
            if existing_admins:
                logger.info(f"Found {len(existing_admins)} existing admin users to delete")
                for admin in existing_admins:
                    db.session.delete(admin)
                db.session.commit()
                logger.info("Successfully deleted existing admin users")
            else:
                logger.info("No existing admin users found to delete")

            # Create a new admin user with the desired password
            admin_email = 'admin@example.com'
            admin_password = 'Abilavz@25'  # This is the password that will be hashed and stored
            
            logger.info(f"Creating new admin user with email: {admin_email}")
            hashed_password = generate_password_hash(admin_password, method='pbkdf2:sha256')
            logger.info("Password hashed successfully")
            
            new_admin = User(email=admin_email, password=hashed_password, role='admin')
            db.session.add(new_admin)
            db.session.commit()
            
            # Verify the user was created
            verify_admin = User.query.filter_by(email=admin_email).first()
            if verify_admin:
                logger.info("Successfully verified new admin user creation")
                logger.info(f"Admin user role: {verify_admin.role}")
                # Test the password hash
                if check_password_hash(verify_admin.password, admin_password):
                    logger.info("Password verification successful")
                else:
                    logger.error("Password verification failed after creation!")
            else:
                logger.error("Failed to verify admin user creation!")
            
            return """
            Admin password has been reset successfully!<br>
            You can now login with:<br>
            Email: admin@example.com<br>
            Password: Abilavz@25<br>
            <br>
            Please try logging in now. If you still have issues, check the server logs.
            """, 200
            
    except Exception as e:
        logger.error(f"Error resetting admin password: {str(e)}")
        return f"Error resetting admin password: {str(e)}", 500

@app.route('/delete_submission/<int:submission_id>', methods=['POST'])
@admin_required
def delete_submission(submission_id):
    # Ensure the user is logged in and is an admin - Handled by @admin_required decorator

    try:
        # Read the CSV file
        if not os.path.exists(USER_SUBMISSIONS_FILE):
            flash('No submissions found.', 'error')
            return redirect(url_for('view_submissions'))

        df = pd.read_csv(USER_SUBMISSIONS_FILE)

        # Check if the submission_id (row index) is valid
        if submission_id < 0 or submission_id >= len(df):
            flash('Submission not found.', 'error')
            return redirect(url_for('view_submissions'))

        # Delete the row with the given submission_id (index)
        df = df.drop(submission_id).reset_index(drop=True) # reset_index to renumber after drop

        # Save the updated dataframe back to CSV
        df.to_csv(USER_SUBMISSIONS_FILE, index=False)

        flash('Submission deleted successfully.', 'success')

    except Exception as e:
        flash(f'Error deleting submission: {str(e)}', 'error')

    return redirect(url_for('view_submissions'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, ssl_context=('cert.pem', 'key.pem'), debug=True)

