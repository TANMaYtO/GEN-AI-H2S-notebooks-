document.addEventListener('DOMContentLoaded', () => {
    // --- Scroll-triggered Animations ---
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-on-scroll').forEach(element => {
        observer.observe(element);
    });

    // --- Form & API Logic ---
    const form = document.getElementById('analysis-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');
    const imageInput = document.getElementById('image-input');
    const fileLabelText = document.getElementById('file-label-text');

    imageInput.addEventListener('change', () => {
        if (imageInput.files.length > 0) {
            fileLabelText.textContent = `File selected: ${imageInput.files[0].name}`;
        } else {
            fileLabelText.textContent = 'Click to Upload an Image';
        }
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const textInput = document.getElementById('text-input').value;
        const imageFile = imageInput.files[0];

        if (!textInput.trim() && !imageFile) {
            alert("Please provide text or an image.");
            return;
        }

        loader.classList.remove('loader-hidden');
        resultsContainer.innerHTML = '';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        // Append data only if it exists to avoid sending empty fields
        if(textInput.trim()) formData.append('text_input', textInput);
        if(imageFile) formData.append('image_file', imageFile);

        try {
            // FIXED: Use the full, absolute URL for the API endpoint
            const response = await fetch('http://127.0.0.1:8080/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || `HTTP error! Status: ${response.status}`);
            }

            // FIXED: Directly use the result object as it's the main data
            displayResults(result);
            
        } catch (error) {
            console.error('Analysis error:', error);
            resultsContainer.innerHTML = `<div class="result-card error-card"><p><strong>Error:</strong> ${error.message}</p></div>`;
        } finally {
            loader.classList.add('loader-hidden');
            analyzeBtn.disabled = false;
        }
    });

    // --- Dynamic Result Display Function ---
    function displayResults(data) {
        resultsContainer.innerHTML = '';
        let delay = 0;
        
        // This function creates a new card and adds it to the DOM with a delay
        const addCard = (card) => {
            if(card) {
                resultsContainer.appendChild(card);
                setTimeout(() => card.classList.add('visible'), delay);
                delay += 200;
            }
        };

        // Handle different report types
        if (data.final_verdict) { // Text-only report
            addCard(createTextCard(data));
        } else { // Image report (which can contain text analysis)
            addCard(createImageManipulationCard(data.image_manipulation_report));
            addCard(createReverseImageSearchCard(data.reverse_image_search_report));
            addCard(createImageContentCard(data.in_image_content_report));
            if (data.extracted_text_analysis_report && data.extracted_text_analysis_report.final_verdict) {
                addCard(createExtractedTextCard(data.extracted_text_analysis_report));
            }
        }
    }
    
    // --- Helper functions to create HTML for each card ---
    function createTextCard(report) {
        if (!report) return null;
        const card = document.createElement('div');
        card.className = 'result-card animate-on-scroll fade-in';
        let sourcesHtml = '';
        if (report.source && Object.keys(report.source).length > 0) {
            sourcesHtml = Object.entries(report.source).map(([source, url]) => 
                `<div class="evidence-item"><strong>${source}</strong><br><a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a></div>`
            ).join('');
        }
        card.innerHTML = `
            <h3><i class="fas fa-file-alt"></i> Text Analysis Report</h3>
            <p><strong>Verdict:</strong> <span class="verdict--${report.final_verdict?.toLowerCase()}">${report.final_verdict || 'N/A'}</span></p>
            <p><strong>Explanation:</strong> ${report.explanation || 'N/A'}</p>
            ${sourcesHtml ? `<div class="sources-section"><h4>Sources:</h4>${sourcesHtml}</div>` : ''}
        `;
        return card;
    }
    
    function createImageManipulationCard(report) {
        if (!report) return null;
        const card = document.createElement('div');
        card.className = 'result-card animate-on-scroll fade-in';
        // FIXED: Use the correct path for the ELA image
        const elaLink = report.ela_image_path ? `<p><strong>ELA Analysis Image:</strong> <a href="/results/${report.ela_image_path}" target="_blank">View ELA Result</a></p>` : '';
        card.innerHTML = `
            <h3><i class="fas fa-shield-alt"></i> Image Manipulation Analysis</h3>
            <p><strong>AI-Generated Score:</strong> ${report.ai_generated_score_percent?.toFixed(2) || '0.00'}%</p>
            <p><strong>Classic Edit Score (ELA):</strong> ${report.classic_edit_score_percent?.toFixed(2) || '0.00'}%</p>
            ${elaLink}
        `;
        return card;
    }
    
    function createImageContentCard(report) {
        if (!report) return null;
        const card = document.createElement('div');
        card.className = 'result-card animate-on-scroll fade-in';
        let contentHtml = '';
        if (report['Safe Search']) {
            const ss = report['Safe Search'];
            contentHtml += `<p><strong>Safe Search:</strong> Adult: ${ss.adult}, Violence: ${ss.violence}</p>`;
        }
        if (report['Identified Entities']) {
            contentHtml += `<p><strong>Identified Entities:</strong> ${report['Identified Entities'].join(', ')}</p>`;
        }
        card.innerHTML = `
            <h3><i class="fas fa-search"></i> Image Content Analysis</h3>
            ${contentHtml || '<p>No specific content detected.</p>'}
        `;
        return card;
    }
    
    function createReverseImageSearchCard(report) {
        if (!report) return null;
        const card = document.createElement('div');
        card.className = 'result-card animate-on-scroll fade-in';
        let matchesHtml = '<p>No online matches found.</p>';
        if (report['Reverse Image Matches'] && report['Reverse Image Matches'].length > 0) {
            matchesHtml = report['Reverse Image Matches'].map(match => 
                `<div class="evidence-item"><strong>${match.title}</strong><br><a href="${match.url}" target="_blank" rel="noopener noreferrer">${match.url}</a></div>`
            ).join('');
        }
        card.innerHTML = `
            <h3><i class="fas fa-globe"></i> Reverse Image Search</h3>
            ${matchesHtml}
        `;
        return card;
    }
    
    function createExtractedTextCard(report) {
        if (!report) return null;
        // Re-use the text card creation logic
        const card = createTextCard(report);
        card.querySelector('h3').innerHTML = `<i class="fas fa-file-text"></i> Extracted Text Analysis`;
        return card;
    }
});
