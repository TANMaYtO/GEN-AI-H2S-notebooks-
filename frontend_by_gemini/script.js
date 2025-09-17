document.addEventListener('DOMContentLoaded', () => {
    // --- Scroll-triggered Animations ---
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Optional: stop observing after it's visible
                // observer.unobserve(entry.target); 
            }
        });
    }, { threshold: 0.1 }); // Trigger when 10% of the element is visible

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

    // Update file input label text
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
        formData.append('text_input', textInput);
        formData.append('image_file', imageFile);

        try {
            // === MOCK BACKEND CALL FOR DEMO ===
            // Replace this with your actual fetch call
            await new Promise(resolve => setTimeout(resolve, 3000)); // Simulate network delay
            const mockData = createMockReport(!!imageFile, !!textInput.trim());
            displayResults(mockData);
            
        } catch (error) {
            resultsContainer.innerHTML = `<div class="result-card"><p><strong>Error:</strong> ${error.message}</p></div>`;
        } finally {
            loader.classList.add('loader-hidden');
            analyzeBtn.disabled = false;
        }
    });

    // --- Dynamic Result Display Function ---
    function displayResults(data) {
        resultsContainer.innerHTML = ''; // Clear container

        // Sequentially fade in each result card for a nice effect
        let delay = 0;
        
        if (data.text_analysis_report) {
            const card = createTextCard(data.text_analysis_report);
            resultsContainer.appendChild(card);
            setTimeout(() => card.classList.add('visible'), delay);
            delay += 200;
        }
        if (data.image_authenticity_report) {
            const card = createImageAuthCard(data.image_authenticity_report);
            resultsContainer.appendChild(card);
            setTimeout(() => card.classList.add('visible'), delay);
            delay += 200;
        }
        if (data.online_history_report) {
            const card = createHistoryCard(data.online_history_report);
            resultsContainer.appendChild(card);
            setTimeout(() => card.classList.add('visible'), delay);
        }
    }
    
    // --- Helper functions to create HTML for each card ---
    function createTextCard(report) {
        const card = document.createElement('div');
        card.className = 'result-card animate-on-scroll fade-in';
        card.innerHTML = `
            <h3><i class="fas fa-file-alt"></i> Text Analysis Report</h3>
            <p><strong>Verdict:</strong> <span class="verdict--${report.final_verdict?.toLowerCase()}">${report.final_verdict || 'N/A'}</span></p>
            <p><strong>Explanation:</strong> ${report.explanation || 'N/A'}</p>
        `;
        return card;
    }
    function createImageAuthCard(report) {
        const card = document.createElement('div');
        card.className = 'result-card animate-on-scroll fade-in';
        card.innerHTML = `
            <h3><i class="fas fa-shield-alt"></i> Image Authenticity Report</h3>
            <p><strong>AI-Generated Score:</strong> ${report.ai_generated_score_percent || '0.00'}%</p>
            <p><strong>Classic Edit Score (ELA):</strong> ${report.classic_edit_score_percent || '0.00'}%</p>
        `;
        return card;
    }
    function createHistoryCard(report) {
         const card = document.createElement('div');
         card.className = 'result-card animate-on-scroll fade-in';
         let matchesHtml = '<p>No online matches found.</p>';
         if(report.matches && report.matches.length > 0) {
             matchesHtml = report.matches.map(match => 
                `<div class="evidence-item"><strong>${match.title}</strong><br><a href="${match.url}" target="_blank">${match.source}</a></div>`
             ).join('');
         }
         card.innerHTML = `
            <h3><i class="fas fa-globe"></i> Online History Report</h3>
            ${matchesHtml}
         `;
         return card;
    }

    // --- Mock Data Function for Demo ---
    function createMockReport(isImage, isText) {
        const report = {};
        if (isText) {
            report.text_analysis_report = {
                final_verdict: "REFUTES",
                explanation: "The provided evidence contradicts the user's claim about the event."
            };
        }
        if (isImage) {
            report.image_authenticity_report = {
                ai_generated_score_percent: "7.84",
                classic_edit_score_percent: "88.12"
            };
            report.online_history_report = {
                matches: [
                    { title: "Fact Check: Viral image from 2015 protest...", source: "factcheck.org", url: "#"},
                    { title: "Old photo resurfaces online...", source: "reuters.com", url: "#"}
                ]
            };
        }
        return report;
    }
});