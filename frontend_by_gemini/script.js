document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analysis-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default page reload

        // Get form data
        const textInput = document.getElementById('text-input').value;
        const imageInput = document.getElementById('image-input').files[0];

        if (!textInput.trim() && !imageInput) {
            alert("Please provide either text or an image to analyze.");
            return;
        }

        // Show loader and disable button
        loader.classList.remove('loader-hidden');
        resultsContainer.innerHTML = ''; // Clear previous results
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';

        // Prepare data to send to the backend
        const formData = new FormData();
        if (textInput.trim()) {
            formData.append('text_input', textInput);
        }
        if (imageInput) {
            formData.append('image_file', imageInput);
        }

        try {
            // === THIS IS WHERE YOU CALL YOUR BACKEND ===
            // Replace '/analyze' with your actual Flask/FastAPI endpoint URL
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            resultsContainer.innerHTML = `<div class="result-card"><p><strong>Error:</strong> ${error.message}</p></div>`;
        } finally {
            // Hide loader and re-enable button
            loader.classList.add('loader-hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
        }
    });

    function displayResults(data) {
        // This function builds the HTML to display the report from your backend
        // You will need to customize this based on the exact JSON structure
        // your Python pipelines return.
        
        let html = '';
        
        // Example for displaying a text analysis report
        if (data.text_analysis_report) {
            const report = data.text_analysis_report;
            html += `
                <div class="result-card">
                    <h3>Text Analysis Report</h3>
                    <p><strong>Verdict:</strong> ${report.final_verdict || 'N/A'}</p>
                    <p><strong>Explanation:</strong> ${report.explanation || 'N/A'}</p>
                </div>
            `;
        }

        // Example for displaying an image analysis report
        if (data.image_analysis_report) {
             const report = data.image_analysis_report;
             html += `
                <div class="result-card">
                    <h3>Image Authenticity Report</h3>
                    <p><strong>AI-Generated Score:</strong> ${report.image_manipulation_report?.ai_generated_score_percent || '0.00'}%</p>
                    <p><strong>Classic Edit Score (ELA):</strong> ${report.image_manipulation_report?.classic_edit_score_percent || '0.00'}%</p>
                </div>
             `;
        }

        resultsContainer.innerHTML = html || `<div class="placeholder"><p>No results to display.</p></div>`;
    }
});