import requests
import json
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

class RouteUtils:
    def __init__(self):
        # OpenRouteService API endpoint (free tier)
        self.ORS_API_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
        # Nominatim API endpoint for geocoding
        self.NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"
        
    def geocode_address(self, address: str) -> Tuple[float, float]:
        """
        Convert address to coordinates using Nominatim
        Appends ", India" to ensure results are within India
        """
        # Append ", India" to the address for better geocoding results
        address_with_country = f"{address}, India"
        
        params = {
            'q': address_with_country,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'in'  # Restrict to India
        }
        headers = {
            'User-Agent': 'RoadAccidentPredictionSystem/1.0'
        }
        
        try:
            response = requests.get(self.NOMINATIM_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data:
                lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                # Verify coordinates are within India's bounding box
                if (6.0 <= lat <= 38.0) and (68.0 <= lon <= 98.0):
                    return lat, lon
                raise ValueError(f"Location {address} is outside India's boundaries")
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
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }
        
        body = {
            "coordinates": [
                [start_coords[1], start_coords[0]],  # OpenRouteService uses [lon, lat]
                [end_coords[1], end_coords[0]]
            ],
            "alternatives": True,
            "instructions": True,
            "preference": "fastest",  # Can be "fastest", "shortest", or "recommended"
            "options": {
                "avoid_features": ["highways", "ferries"],  # Avoid highways and ferries
                "avoid_countries": ["PK", "CN", "BD", "NP", "BT", "MM"]  # Avoid crossing borders
            }
        }
        
        try:
            response = requests.post(
                self.ORS_API_URL,
                headers=headers,
                json=body
            )
            response.raise_for_status()
            return response.json()['routes']
        except Exception as e:
            raise Exception(f"Route calculation error: {str(e)}")

    def calculate_route_safety_score(self, route_coordinates: List[List[float]], 
                                   accident_data: Dict) -> float:
        """
        Calculate safety score for a route based on accident data
        Returns a score between 0 (safest) and 1 (most dangerous)
        """
        if not accident_data:
            return 0.5  # Default score if no accident data available
            
        score = 0.0
        total_segments = len(route_coordinates) - 1
        
        for i in range(total_segments):
            segment_start = route_coordinates[i]
            segment_end = route_coordinates[i + 1]
            
            # Check if this segment passes through any accident hotspots
            segment_score = self._check_segment_safety(segment_start, segment_end, accident_data)
            score += segment_score
            
        return min(score / total_segments, 1.0)

    def _check_segment_safety(self, start: List[float], end: List[float], 
                            accident_data: Dict) -> float:
        """
        Check safety of a route segment based on accident data
        """
        # Calculate the bounding box for this segment
        min_lat = min(start[0], end[0])
        max_lat = max(start[0], end[0])
        min_lon = min(start[1], end[1])
        max_lon = max(start[1], end[1])
        
        # Find accidents within this bounding box
        nearby_accidents = [
            acc for acc in accident_data
            if (min_lat <= float(acc.get('Latitude', 0)) <= max_lat and
                min_lon <= float(acc.get('Longitude', 0)) <= max_lon)
        ]
        
        if not nearby_accidents:
            return 0.2  # Safe score if no accidents in this area
            
        # Calculate severity-weighted score
        severity_weights = {
            'Fatal': 1.0,
            'Serious': 0.6,
            'Slight': 0.2
        }
        
        total_weight = sum(
            severity_weights.get(acc.get('Accident_Severity', 'Slight'), 0.2)
            for acc in nearby_accidents
        )
        
        # Normalize score based on number of accidents
        return min(total_weight / len(nearby_accidents), 1.0)

    def get_safest_route(self, routes: List[Dict], accident_data: Dict) -> Dict:
        """
        Determine the safest route from available alternatives
        """
        scored_routes = []
        
        for route in routes:
            coordinates = [[point[1], point[0]] for point in route['geometry']['coordinates']]
            safety_score = self.calculate_route_safety_score(coordinates, accident_data)
            
            scored_routes.append({
                'route': route,
                'safety_score': safety_score
            })
        
        # Sort routes by safety score (ascending - lower is safer)
        scored_routes.sort(key=lambda x: x['safety_score'])
        return scored_routes[0]['route'] 