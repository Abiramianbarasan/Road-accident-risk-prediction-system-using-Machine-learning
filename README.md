# Road Accident risk Prediction system using Machine Learning

## 1. Project Overview
This project aims to predict and classify road accident severity using machine learning techniques. The system helps in early assessment of accident severity, enabling faster response and better resource allocation for emergency services.

## 2. System Architecture

### 2.1 Components
- **Frontend**: Web-based user interface (HTML, CSS, JavaScript)
- **Backend**: Flask web application
- **Machine Learning Model**: Random Forest Classifier
- **Data Storage**: CSV file for training data
- **Alert System**: SMS notification system

### 2.2 Technology Stack
- **Programming Language**: Python 3.6
- **Web Framework**: Flask
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Frontend**: HTML5, CSS3, JavaScript

## 3. Features

### 3.1 Core Features
1. **Accident Severity Prediction**
   - Input collection for accident parameters
   - Real-time severity prediction
   - Three-level classification (Fatal, Serious, Slight)

2. **User Interface**
   - Interactive web form
   - Real-time results display
   - Responsive design

3. **Alert System**
   - SMS notifications
   - Customizable alert messages
   - Multiple recipient support

### 3.2 Input Parameters
1. **Police Attendance**
   - Did Police Officer Attend Scene of Accident (Yes/No)

2. **Environmental Conditions**
   - Light Conditions
     - Daylight
     - Darkness - lights lit
     - Darkness - lights unlit
     - Darkness - no lighting
     - Fog or mist

   - Road Surface Conditions
     - Dry
     - Wet or damp
     - Snow
     - Frost or Ice
     - Flood
     - Mud

   - Weather Conditions
     - Fine no high winds
     - Raining no high winds
     - Snowing no high winds
     - Fine + high winds
     - Raining + high winds
     - Snowing + high winds
     - Fog or mist

3. **Speed Parameters**
   - Speed Limit (5-120 km/h)

## 4. Machine Learning Model

### 4.1 Model Details
- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 300
  - max_depth: 15
  - min_samples_split: 3
  - min_samples_leaf: 1
  - criterion: entropy
  - class_weight: balanced

### 4.2 Training Data
- Source: cleaned_accident_data_preprocessed (4).csv
- Features:
  - Did_Police_Officer_Attend_Scene_of_Accident
  - Light_Conditions
  - Road_Surface_Conditions
  - Speed_limit
  - Weather_Conditions

### 4.3 Model Performance
- Accuracy: [To be calculated]
- Precision: [To be calculated]
- Recall: [To be calculated]
- F1-Score: [To be calculated]

## 5. Implementation Details

### 5.1 Data Preprocessing
1. **Data Cleaning**
   - Handling missing values
   - Removing duplicates
   - Standardizing formats

2. **Feature Engineering**
   - Categorical encoding
   - Feature scaling
   - Feature selection

### 5.2 Model Training
1. **Training Process**
   - Data splitting (train/test)
   - Cross-validation
   - Hyperparameter tuning
   - Model evaluation

2. **Model Persistence**
   - Model saving (joblib)
   - Model loading
   - Version control

### 5.3 Web Application
1. **Routes**
   - Home page (/)
   - Prediction endpoint (/)
   - SMS alert endpoint (/sms/)
   - Visualization endpoint (/visual/)

2. **Templates**
   - index.html (Main form)
   - visual.html (Statistics visualization)

## 6. Testing and Validation

### 6.1 Test Cases
1. **Fatal Accident Scenarios**
   - High speed with poor conditions
   - Night time with wet road
   - Snowy conditions
   - Foggy conditions
   - Flooded road

2. **Serious Accident Scenarios**
   - Moderate speed with poor visibility
   - Wet road with moderate speed
   - Dusk conditions
   - Light rain
   - Moderate wind

3. **Slight Accident Scenarios**
   - Good conditions with low speed
   - Urban area with low speed
   - Residential area
   - School zone
   - Parking lot

### 6.2 Edge Cases
1. No police officer present
2. Extreme weather conditions
3. Minimum speed limit
4. Maximum speed limit
5. Mixed conditions

## 7. Future Enhancements

### 7.1 Planned Features
1. **Advanced Analytics**
   - Real-time accident pattern analysis
   - Geographic heat maps
   - Time-based trend analysis

2. **Integration**
   - Emergency services API integration
   - Traffic management system integration
   - Weather API integration

3. **User Experience**
   - Mobile application
   - Voice input support
   - Multi-language support

### 7.2 Model Improvements
1. **Algorithm Enhancement**
   - Deep learning integration
   - Ensemble methods
   - Real-time learning

2. **Data Enhancement**
   - Additional data sources
   - Real-time data integration
   - Historical data analysis

## 8. Deployment and Maintenance

### 8.1 System Requirements
- Python 3.x
- Required packages (requirements.txt):
  - flask==2.0.1
  - pandas==1.3.3
  - scikit-learn==0.24.2
  - numpy==1.21.2

### 8.2 Installation Steps
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

### 8.3 Maintenance
- Regular model retraining
- Data quality checks
- Performance monitoring
- Security updates

## 9. Conclusion
This project demonstrates the effective use of machine learning in predicting road accident severity. The system provides valuable insights for emergency response teams and helps in better resource allocation. Future enhancements will focus on improving accuracy and expanding the system's capabilities.

## 10. References
1. scikit-learn documentation
2. Flask documentation
3. Road accident datasets
4. Machine learning research papers 
