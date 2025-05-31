import requests
import json
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import joblib
import os

class RouteUtils:
    def __init__(self):
        # OpenRouteService API endpoint (free tier)
        self.ORS_API_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
        # Nominatim API endpoint for geocoding
        self.NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"
        
        # Load the trained accident severity prediction model
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_model_india.sav')
            self.severity_model = joblib.load(model_path)
            print("Successfully loaded accident severity prediction model")
        except Exception as e:
            print(f"Warning: Could not load accident severity model: {str(e)}")
            self.severity_model = None
        
    def geocode_address(self, address: str) -> Tuple[float, float]:
        """
        Convert address to coordinates using Nominatim
        """
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'RoadAccidentPredictionSystem/1.0'
        }
        
        try:
            response = requests.get(self.NOMINATIM_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
            raise ValueError(f"Could not geocode address: {address}")
        except Exception as e:
            raise Exception(f"Geocoding error: {str(e)}")

    def get_route_alternatives(self, start_coords: Tuple[float, float], 
                             end_coords: Tuple[float, float],
                             api_key: str) -> List[Dict]:
        """
        Get multiple route alternatives using OpenRouteService
        """
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        if not (-90 <= start_coords[0] <= 90 and -180 <= start_coords[1] <= 180):
            raise ValueError(f"Invalid start coordinates: {start_coords}")
        if not (-90 <= end_coords[0] <= 90 and -180 <= end_coords[1] <= 180):
            raise ValueError(f"Invalid end coordinates: {end_coords}")
        
        body = {
            "coordinates": [
                [float(start_coords[1]), float(start_coords[0])],
                [float(end_coords[1]), float(end_coords[0])]
            ],
            "alternatives": True,
            "instructions": True
        }
        
        try:
            print(f"DEBUG: Sending coordinates to OpenRouteService: {body['coordinates']}")
            response = requests.post(
                self.ORS_API_URL,
                headers=headers,
                json=body
            )
            
            if response.status_code != 200:
                error_msg = f"OpenRouteService API error: {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
                
            return response.json()['routes']
        except requests.exceptions.RequestException as e:
            error_msg = f"Route calculation error: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    def calculate_route_safety_score(self, route_coordinates: List[List[float]], 
                                   accident_data: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive safety score for a route based on accident data
        Returns a dictionary containing:
        - overall_score: Score between 0 (safest) and 1 (most dangerous)
        - accident_density: Number of accidents per km
        - severity_score: Weighted score based on accident severity
        - risk_zones: List of high-risk segments
        """
        if not route_coordinates or len(route_coordinates) < 2:
            raise ValueError("Invalid route coordinates")

        total_distance = 0
        accident_count = 0
        severity_sum = 0
        risk_zones = []
        
        # Calculate total route distance and analyze segments
        for i in range(len(route_coordinates) - 1):
            segment_start = route_coordinates[i]
            segment_end = route_coordinates[i + 1]
            
            # Calculate segment distance (in kilometers)
            segment_distance = self._calculate_distance(segment_start, segment_end)
            total_distance += segment_distance
            
            # Analyze segment safety
            segment_analysis = self._analyze_segment_safety(
                segment_start, 
                segment_end, 
                accident_data
            )
            
            accident_count += segment_analysis['accident_count']
            severity_sum += segment_analysis['severity_score']
            
            # Track high-risk zones (segments with accidents)
            if segment_analysis['accident_count'] > 0:
                risk_zones.append({
                    'start': segment_start,
                    'end': segment_end,
                    'accident_count': segment_analysis['accident_count'],
                    'severity': segment_analysis['severity_score']
                })
        
        # Calculate safety metrics
        accident_density = accident_count / total_distance if total_distance > 0 else 0
        severity_score = severity_sum / len(route_coordinates) if route_coordinates else 0
        
        # Calculate overall safety score (weighted combination of factors)
        overall_score = self._calculate_overall_safety_score(
            accident_density,
            severity_score,
            total_distance
        )
        
        return {
            'overall_score': min(overall_score, 1.0),  # Cap at 1.0
            'accident_density': accident_density,
            'severity_score': severity_score,
            'total_distance': total_distance,
            'accident_count': accident_count,
            'risk_zones': risk_zones
        }

    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """
        Calculate distance between two points using Haversine formula
        Returns distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = radians(point1[0]), radians(point1[1])
        lat2, lon2 = radians(point2[0]), radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance

    def _analyze_segment_safety(self, start: List[float], end: List[float], 
                              accident_data: Dict) -> Dict:
        """
        Analyze safety of a route segment based on accident data
        Returns dictionary with accident count and severity score
        """
        accident_count = 0
        severity_score = 0.0
        
        # Define search radius (in kilometers)
        SEARCH_RADIUS = 0.5  # 500 meters
        
        # Get accidents within the search radius of the segment
        nearby_accidents = self._find_nearby_accidents(
            start, end, accident_data, SEARCH_RADIUS
        )
        
        for accident in nearby_accidents:
            accident_count += 1
            # Weight severity based on accident type and severity
            severity = accident.get('severity', 1.0)  # Default to 1.0 if not specified
            severity_score += severity
        
        return {
            'accident_count': accident_count,
            'severity_score': severity_score / max(accident_count, 1)
        }

    def _find_nearby_accidents(self, start: List[float], end: List[float],
                             accident_data: Dict, radius: float) -> List[Dict]:
        """
        Find accidents near a route segment
        Returns list of accidents within the specified radius
        """
        nearby_accidents = []
        
        for accident in accident_data.get('accidents', []):
            acc_lat = accident.get('latitude')
            acc_lon = accident.get('longitude')
            
            if acc_lat is None or acc_lon is None:
                continue
                
            # Check if accident is within radius of segment
            if self._is_point_near_segment(
                [acc_lat, acc_lon],
                start,
                end,
                radius
            ):
                nearby_accidents.append(accident)
        
        return nearby_accidents

    def _is_point_near_segment(self, point: List[float], start: List[float],
                             end: List[float], radius: float) -> bool:
        """
        Check if a point is within radius kilometers of a line segment
        """
        # Calculate distance from point to line segment
        distance = self._point_to_segment_distance(point, start, end)
        return distance <= radius

    def _point_to_segment_distance(self, point: List[float], start: List[float],
                                 end: List[float]) -> float:
        """
        Calculate minimum distance from a point to a line segment
        Returns distance in kilometers
        """
        # Convert to radians
        lat1, lon1 = radians(point[0]), radians(point[1])
        lat2, lon2 = radians(start[0]), radians(start[1])
        lat3, lon3 = radians(end[0]), radians(end[1])
        
        # Calculate distance using great circle formula
        R = 6371  # Earth's radius in kilometers
        
        # Calculate distances
        d1 = self._calculate_distance(point, start)
        d2 = self._calculate_distance(point, end)
        d3 = self._calculate_distance(start, end)
        
        # If segment is very short, return distance to closest endpoint
        if d3 < 0.001:  # Less than 1 meter
            return min(d1, d2)
        
        # Calculate perpendicular distance
        s = (d1 + d2 + d3) / 2  # Semi-perimeter
        area = sqrt(s * (s - d1) * (s - d2) * (s - d3))
        height = 2 * area / d3
        
        return height

    def _calculate_overall_safety_score(self, accident_density: float,
                                     severity_score: float,
                                     total_distance: float) -> float:
        """
        Calculate overall safety score based on multiple factors
        Returns score between 0 (safest) and 1 (most dangerous)
        """
        # Weights for different factors
        DENSITY_WEIGHT = 0.4
        SEVERITY_WEIGHT = 0.4
        DISTANCE_WEIGHT = 0.2
        
        # Normalize factors
        normalized_density = min(accident_density / 10, 1.0)  # Cap at 10 accidents per km
        normalized_severity = min(severity_score / 5, 1.0)    # Cap at severity 5
        normalized_distance = min(total_distance / 50, 1.0)   # Cap at 50 km
        
        # Calculate weighted score
        score = (
            DENSITY_WEIGHT * normalized_density +
            SEVERITY_WEIGHT * normalized_severity +
            DISTANCE_WEIGHT * (1 - normalized_distance)  # Longer routes are considered safer
        )
        
        return score

    def get_safest_route(self, routes: List[Dict], accident_data: Dict) -> Dict:
        """
        Determine the safest route from available alternatives
        Returns the route with the lowest safety score and detailed safety analysis
        """
        scored_routes = []
        
        for route in routes:
            coordinates = [[point[1], point[0]] for point in route['geometry']['coordinates']]
            safety_analysis = self.calculate_route_safety_score(coordinates, accident_data)
            
            scored_routes.append({
                'route': route,
                'safety_analysis': safety_analysis
            })
        
        # Sort routes by overall safety score (ascending - lower is safer)
        scored_routes.sort(key=lambda x: x['safety_analysis']['overall_score'])
        
        # Return the safest route with its analysis
        safest_route = scored_routes[0]
        return {
            'route': safest_route['route'],
            'safety_analysis': safest_route['safety_analysis'],
            'alternative_count': len(routes),
            'safety_rank': 1
        }

    def predict_accident_severity(self, features: Dict) -> str:
        """
        Predict accident severity using the trained model
        Returns: 'Fatal', 'Serious', or 'Slight'
        """
        if not self.severity_model:
            return 'Slight'  # Default if model not loaded
            
        try:
            # Prepare features in the correct order
            feature_vector = np.array([
                float(features.get('Did_Police_Officer_Attend', 1)),
                float(features.get('Light_Conditions', 1)),
                float(features.get('Road_Surface_Conditions', 1)),
                float(features.get('Speed_limit', 30)),
                float(features.get('Weather_Conditions', 1)),
                float(features.get('Latitude', 0)),
                float(features.get('Longitude', 0))
            ]).reshape(1, -1)
            
            # Get prediction and probability
            prediction = self.severity_model.predict(feature_vector)[0]
            probabilities = self.severity_model.predict_proba(feature_vector)[0]
            
            # Map prediction to severity level
            severity_map = {0: 'Fatal', 1: 'Serious', 2: 'Slight'}
            predicted_severity = severity_map.get(prediction, 'Slight')
            
            return {
                'severity': predicted_severity,
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'Fatal': float(probabilities[0]),
                    'Serious': float(probabilities[1]),
                    'Slight': float(probabilities[2])
                }
            }
        except Exception as e:
            print(f"Error predicting accident severity: {str(e)}")
            return {'severity': 'Slight', 'confidence': 0.0, 'probabilities': {'Fatal': 0.0, 'Serious': 0.0, 'Slight': 1.0}}

    def analyze_route_safety(self, route_coordinates: List[List[float]], 
                           current_conditions: Dict) -> Dict:
        """
        Comprehensive route safety analysis combining historical data and current conditions
        Returns detailed safety analysis including predicted accident severity
        """
        if not route_coordinates or len(route_coordinates) < 2:
            raise ValueError("Invalid route coordinates")

        # Get historical safety analysis
        historical_analysis = self.calculate_route_safety_score(route_coordinates, current_conditions)
        
        # Analyze current conditions along the route
        current_conditions_analysis = []
        risk_segments = []
        
        # Sample points along the route for current conditions analysis
        for i in range(0, len(route_coordinates) - 1, 5):  # Sample every 5th point
            point = route_coordinates[i]
            
            # Prepare features for severity prediction
            features = {
                'Did_Police_Officer_Attend': current_conditions.get('Did_Police_Officer_Attend', 1),
                'Light_Conditions': current_conditions.get('Light_Conditions', 1),
                'Road_Surface_Conditions': current_conditions.get('Road_Surface_Conditions', 1),
                'Speed_limit': current_conditions.get('Speed_limit', 30),
                'Weather_Conditions': current_conditions.get('Weather_Conditions', 1),
                'Latitude': point[0],
                'Longitude': point[1]
            }
            
            # Get severity prediction for this point
            severity_prediction = self.predict_accident_severity(features)
            
            # If high risk, add to risk segments
            if severity_prediction['severity'] in ['Fatal', 'Serious']:
                risk_segments.append({
                    'location': point,
                    'severity': severity_prediction['severity'],
                    'confidence': severity_prediction['confidence']
                })
            
            current_conditions_analysis.append({
                'location': point,
                'prediction': severity_prediction
            })
        
        # Calculate overall risk score combining historical and current conditions
        historical_weight = 0.6
        current_weight = 0.4
        
        # Calculate current conditions risk score
        current_risk_score = sum(
            1.0 if pred['prediction']['severity'] == 'Fatal'
            else 0.6 if pred['prediction']['severity'] == 'Serious'
            else 0.2
            for pred in current_conditions_analysis
        ) / len(current_conditions_analysis)
        
        # Combine scores
        overall_risk_score = (
            historical_weight * historical_analysis['overall_score'] +
            current_weight * current_risk_score
        )
        
        return {
            'historical_analysis': historical_analysis,
            'current_conditions_analysis': current_conditions_analysis,
            'risk_segments': risk_segments,
            'overall_risk_score': min(overall_risk_score, 1.0),
            'risk_level': 'High' if overall_risk_score > 0.7 else 'Medium' if overall_risk_score > 0.4 else 'Low',
            'recommendations': self._generate_safety_recommendations(
                overall_risk_score,
                risk_segments,
                current_conditions
            )
        }

    def _generate_safety_recommendations(self, risk_score: float,
                                      risk_segments: List[Dict],
                                      current_conditions: Dict) -> List[str]:
        """
        Generate safety recommendations based on risk analysis
        """
        recommendations = []
        
        # High risk recommendations
        if risk_score > 0.7:
            recommendations.extend([
                "⚠️ HIGH RISK ROUTE - Consider alternative route",
                "Exercise extreme caution if this route must be taken",
                "Ensure vehicle is in optimal condition",
                "Consider postponing travel if possible"
            ])
        
        # Medium risk recommendations
        elif risk_score > 0.4:
            recommendations.extend([
                "⚠️ MEDIUM RISK ROUTE - Proceed with caution",
                "Maintain safe following distance",
                "Be extra vigilant at intersections",
                "Consider reducing speed in high-risk segments"
            ])
        
        # Low risk recommendations
        else:
            recommendations.extend([
                "✅ LOW RISK ROUTE - Standard safety precautions apply",
                "Maintain normal speed limits",
                "Stay alert and follow traffic rules"
            ])
        
        # Add condition-specific recommendations
        if current_conditions.get('Weather_Conditions') in [2, 3, 4, 5, 6, 7]:  # Adverse weather
            recommendations.append("⚠️ Adverse weather conditions - Reduce speed and increase following distance")
        
        if current_conditions.get('Light_Conditions') in [2, 3, 4, 5, 6]:  # Low light conditions
            recommendations.append("⚠️ Low visibility conditions - Use headlights and reduce speed")
        
        if current_conditions.get('Road_Surface_Conditions') in [2, 3, 4, 5, 6]:  # Poor road conditions
            recommendations.append("⚠️ Poor road conditions - Reduce speed and avoid sudden maneuvers")
        
        # Add specific segment warnings
        for segment in risk_segments:
            if segment['severity'] == 'Fatal':
                recommendations.append(
                    f"⚠️ HIGH RISK SEGMENT at coordinates {segment['location']} - "
                    f"Exercise extreme caution (Confidence: {segment['confidence']:.0%})"
                )
        
        return recommendations