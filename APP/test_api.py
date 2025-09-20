"""
Test script for Digital Forensics Toolkit API
This script helps you test both text and image analysis endpoints.
"""

import requests
import json
import os
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

def test_health():
    """Test the health endpoint to check if server is running."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy!")
            print(f"   - Initialized: {data.get('initialized', False)}")
            print(f"   - Models loaded: {data.get('models_loaded', 0)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the server is running with: python main.py")
        return False

def test_text_analysis():
    """Test text analysis with sample claims."""
    print("\n📝 Testing text analysis...")
    
    test_claims = [
        "The Earth is flat.",
        "COVID-19 vaccines are safe and effective.",
        "The moon landing was faked.",
        "Climate change is real and caused by human activities."
    ]
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n--- Test {i}: '{claim}' ---")
        
        try:
            # Prepare form data
            form_data = {
                'text_input': claim
            }
            
            # Make request
            response = requests.post(ANALYZE_ENDPOINT, data=form_data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result.get('data', {})
                    text_report = data.get('text_analysis_report', {})
                    
                    print(f"✅ Analysis completed!")
                    print(f"   Verdict: {text_report.get('final_verdict', 'N/A')}")
                    print(f"   Explanation: {text_report.get('explanation', 'N/A')[:100]}...")
                    
                    if text_report.get('source'):
                        print(f"   Sources: {len(text_report['source'])} found")
                else:
                    print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")

def test_image_analysis():
    """Test image analysis with sample images."""
    print("\n🖼️ Testing image analysis...")
    
    # Look for test images in common locations
    test_image_paths = [
        "test_image.jpg",
        "test_image.png", 
        "sample.jpg",
        "sample.png",
        "image.jpg",
        "image.png"
    ]
    
    # Check current directory and common image locations
    found_images = []
    for path in test_image_paths:
        if os.path.exists(path):
            found_images.append(path)
    
    if not found_images:
        print("❌ No test images found!")
        print("   Please add a test image (JPG/PNG) to the current directory")
        print("   You can use any image file for testing")
        return
    
    for image_path in found_images[:2]:  # Test up to 2 images
        print(f"\n--- Testing image: {image_path} ---")
        
        try:
            # Prepare file upload
            with open(image_path, 'rb') as image_file:
                files = {
                    'image_file': (image_path, image_file, 'image/jpeg')
                }
                
                # Make request
                response = requests.post(ANALYZE_ENDPOINT, files=files, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result.get('data', {})
                    
                    print(f"✅ Image analysis completed!")
                    
                    # Display manipulation report
                    if 'image_manipulation_report' in data:
                        manip_report = data['image_manipulation_report']
                        print(f"   AI Generated Score: {manip_report.get('ai_generated_score_percent', 'N/A')}%")
                        print(f"   Classic Edit Score: {manip_report.get('classic_edit_score_percent', 'N/A')}%")
                    
                    # Display content report
                    if 'in_image_content_report' in data:
                        content_report = data['in_image_content_report']
                        if content_report.get('Extracted Text'):
                            text = content_report['Extracted Text'][:100]
                            print(f"   Extracted Text: {text}...")
                    
                    # Display reverse search results
                    if 'reverse_image_search_report' in data:
                        search_report = data['reverse_image_search_report']
                        matches = search_report.get('Reverse Image Matches', [])
                        print(f"   Reverse Search: {len(matches)} matches found")
                    
                    # Display text analysis if text was extracted
                    if 'extracted_text_analysis_report' in data:
                        text_report = data['extracted_text_analysis_report']
                        print(f"   Text Analysis Verdict: {text_report.get('final_verdict', 'N/A')}")
                        
                else:
                    print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except FileNotFoundError:
            print(f"❌ Image file not found: {image_path}")

def test_web_interface():
    """Test the web interface."""
    print("\n🌐 Testing web interface...")
    
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            print("✅ Web interface is accessible!")
            print(f"   Open your browser and go to: {BASE_URL}")
        else:
            print(f"❌ Web interface error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot access web interface: {e}")

def main():
    """Run all tests."""
    print("🚀 Digital Forensics Toolkit - API Testing")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Server is not running. Please start it first with: python main.py")
        return
    
    # Test 2: Web interface
    test_web_interface()
    
    # Test 3: Text analysis
    test_text_analysis()
    
    # Test 4: Image analysis
    test_image_analysis()
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print(f"🌐 Web interface: {BASE_URL}")
    print("📚 API documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
