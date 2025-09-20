"""
Sample test data for Digital Forensics Toolkit
"""

# Sample text claims for testing
SAMPLE_CLAIMS = [
    "The Earth is flat and NASA is lying about it.",
    "COVID-19 vaccines contain microchips for tracking people.",
    "The moon landing in 1969 was filmed in a Hollywood studio.",
    "Climate change is a hoax created by scientists for funding.",
    "5G networks cause cancer and spread COVID-19.",
    "The Great Wall of China is visible from space with the naked eye.",
    "Humans only use 10% of their brain capacity.",
    "Sharks don't get cancer, so eating shark cartilage prevents cancer.",
    "The human body has 206 bones.",
    "Water boils at 100 degrees Celsius at sea level."
]

# Instructions for testing
TESTING_INSTRUCTIONS = """
🧪 DIGITAL FORENSICS TOOLKIT - TESTING GUIDE

1. 🚀 START THE SERVER
   - Open terminal/command prompt
   - Navigate to your project directory: cd "E:\GENAI H2S\APP"
   - Run: python main.py
   - Wait for "✅ All models and data loaded successfully!" message
   - Server will be available at: http://localhost:8000

2. 🌐 TEST WEB INTERFACE
   - Open browser and go to: http://localhost:8000
   - You should see the Digital Forensics Toolkit interface
   - Try uploading text or images through the web form

3. 🔧 TEST API ENDPOINTS
   - Run the test script: python test_api.py
   - This will test all endpoints automatically
   - Check the output for any errors

4. 📝 TEST TEXT ANALYSIS
   - Use the sample claims provided above
   - Or test with your own text/claims
   - Check the verdict and explanation

5. 🖼️ TEST IMAGE ANALYSIS
   - Add any JPG/PNG image to your project directory
   - Name it something like "test_image.jpg"
   - Upload through web interface or API
   - Check manipulation scores and extracted text

6. 📊 CHECK RESULTS
   - Text analysis should show: verdict, explanation, sources
   - Image analysis should show: manipulation scores, OCR text, reverse search
   - All results should be displayed in the web interface

7. 🐛 TROUBLESHOOTING
   - If server won't start: Check if all dependencies are installed
   - If models fail to load: Check GPU memory or use CPU fallback
   - If API errors: Check the server logs for detailed error messages
   - If frontend not loading: Check if static files are in correct location

8. 📚 API DOCUMENTATION
   - Visit: http://localhost:8000/docs
   - This shows the interactive API documentation
   - You can test endpoints directly from the browser

9. 🔍 HEALTH CHECK
   - Visit: http://localhost:8000/health
   - This shows server status and loaded models
   - Useful for debugging initialization issues
"""

print(TESTING_INSTRUCTIONS)
print("\n📋 SAMPLE CLAIMS FOR TESTING:")
for i, claim in enumerate(SAMPLE_CLAIMS, 1):
    print(f"{i:2d}. {claim}")
