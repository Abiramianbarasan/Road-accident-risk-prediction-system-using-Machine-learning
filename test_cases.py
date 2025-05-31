def test_route_safety_analysis():
    print("\nTesting Route Safety Analysis...\n")
    
    from utils.route_utils import RouteUtils
    route_utils = RouteUtils()
    
    # Test cases for route safety analysis
    test_cases = [
        {
            'name': 'High Risk Route',
            'route': [
                [22.9734, 78.6569],  # Start point
                [22.9834, 78.6669],  # High risk segment
                [22.9934, 78.6769]   # End point
            ],
            'conditions': {
                'Did_Police_Officer_Attend': 1,
                'Light_Conditions': 6,  # Darkness - no lighting
                'Road_Surface_Conditions': 5,  # Flood
                'Speed_limit': 70,
                'Weather_Conditions': 7  # Fog or mist
            },
            'expected_risk_level': 'High'
        },
        {
            'name': 'Medium Risk Route',
            'route': [
                [22.9734, 78.6569],
                [22.9834, 78.6669],
                [22.9934, 78.6769]
            ],
            'conditions': {
                'Did_Police_Officer_Attend': 1,
                'Light_Conditions': 4,  # Darkness - lights lit
                'Road_Surface_Conditions': 2,  # Wet or damp
                'Speed_limit': 40,
                'Weather_Conditions': 2  # Raining no high winds
            },
            'expected_risk_level': 'Medium'
        },
        {
            'name': 'Low Risk Route',
            'route': [
                [22.9734, 78.6569],
                [22.9834, 78.6669],
                [22.9934, 78.6769]
            ],
            'conditions': {
                'Did_Police_Officer_Attend': 1,
                'Light_Conditions': 1,  # Daylight
                'Road_Surface_Conditions': 1,  # Dry
                'Speed_limit': 30,
                'Weather_Conditions': 1  # Fine no high winds
            },
            'expected_risk_level': 'Low'
        }
    ]
    
    total_cases = len(test_cases)
    passed_cases = 0
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Conditions: {test_case['conditions']}")
        
        try:
            # Analyze route safety
            analysis = route_utils.analyze_route_safety(
                test_case['route'],
                test_case['conditions']
            )
            
            # Check risk level
            actual_risk_level = analysis['risk_level']
            expected_risk_level = test_case['expected_risk_level']
            
            print(f"Expected Risk Level: {expected_risk_level}")
            print(f"Actual Risk Level: {actual_risk_level}")
            print(f"Overall Risk Score: {analysis['overall_risk_score']:.2f}")
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"- {rec}")
            
            if actual_risk_level == expected_risk_level:
                print("PASSED")
                passed_cases += 1
            else:
                print("FAILED - Risk level mismatch")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
    
    # Print summary
    print("\nRoute Safety Analysis Test Summary:")
    print(f"Total Cases: {total_cases}")
    print(f"Passed: {passed_cases}")
    print(f"Failed: {total_cases - passed_cases}")
    print(f"Success Rate: {(passed_cases/total_cases)*100:.2f}%")

if __name__ == "__main__":
    run_tests()
    test_route_safety_analysis() 