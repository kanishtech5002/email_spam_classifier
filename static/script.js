// ==========================================================================
// Email Spam Classifier - Interactive Frontend Scripting
// ==========================================================================

// Track character count dynamically
const emailTextarea = document.getElementById('email-text');
const charCountSpan = document.getElementById('char-count');

if (emailTextarea) {
    emailTextarea.addEventListener('input', () => {
        charCountSpan.textContent = emailTextarea.value.length;
    });
}

/**
 * Predicts whether the input message is Spam or Ham.
 * Calls Flask backend prediction API.
 */
async function predictSpam() {
    const text = emailTextarea.value.trim();
    
    // UI Elements
    const btnPredict = document.getElementById('btn-predict');
    const btnClear = document.getElementById('btn-clear');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');
    const resultPrediction = document.getElementById('result-prediction');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceBar = document.getElementById('confidence-bar');
    const processedTokens = document.getElementById('processed-tokens');

    // Input Validation
    if (!text) {
        alert("Please enter some email or SMS text to predict.");
        return;
    }

    // Prepare UI states: Show loader, disable buttons, hide old results
    loader.classList.remove('hidden');
    resultContainer.classList.add('hidden');
    btnPredict.disabled = true;
    btnClear.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        // Ensure the response is JSON before parsing to avoid syntax error crashes
        const contentType = response.headers.get('content-type') || '';
        let data;

        if (contentType.includes('application/json')) {
            data = await response.json();
        } else {
            const rawText = await response.text();
            throw new Error(`Server returned unexpected non-JSON response (${response.status}): ${rawText.substring(0, 100)}`);
        }

        // Hide loader and re-enable buttons
        loader.classList.add('hidden');
        btnPredict.disabled = false;
        btnClear.disabled = false;

        if (response.ok && data.success) {
            // Configure UI for Result displays
            resultContainer.classList.remove('hidden');
            
            // Clean up old classes
            resultCard.className = 'result-card';
            
            // Render predictions and confidence
            resultPrediction.textContent = data.prediction;
            resultConfidence.textContent = `${data.confidence}%`;
            
            // Reset progress bar width first to allow transition effect
            confidenceBar.style.width = '0%';
            
            // Display clean tokens
            processedTokens.textContent = data.processed_text || '(No tokens remaining after filtering)';

            if (data.prediction === 'Spam') {
                resultCard.classList.add('spam');
                resultIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color: #ef4444;"></i>';
                // Trigger transition delay for bar filling animation
                setTimeout(() => {
                    confidenceBar.style.width = `${data.confidence}%`;
                }, 100);
            } else {
                resultCard.classList.add('ham');
                resultIcon.innerHTML = '<i class="fa-solid fa-circle-check" style="color: #10b981;"></i>';
                setTimeout(() => {
                    confidenceBar.style.width = `${data.confidence}%`;
                }, 100);
            }
        } else {
            const errorMsg = data && data.error ? data.error : `Request failed with status ${response.status}`;
            alert(`Error: ${errorMsg}`);
        }

    } catch (error) {
        loader.classList.add('hidden');
        btnPredict.disabled = false;
        btnClear.disabled = false;
        alert(`Failed to connect to classifier API: ${error.message}`);
    }
}

/**
 * Clears the textarea and prediction results.
 */
function clearInput() {
    emailTextarea.value = '';
    charCountSpan.textContent = 0;
    
    const resultContainer = document.getElementById('result-container');
    const confidenceBar = document.getElementById('confidence-bar');
    
    if (resultContainer) {
        resultContainer.classList.add('hidden');
    }
    if (confidenceBar) {
        confidenceBar.style.width = '0%';
    }
}

/**
 * Tab-switching controller for analytics charts.
 * 
 * @param {Event} event - The click event triggers this.
 * @param {string} tabId - The container ID of the tab pane.
 */
function switchTab(event, tabId) {
    // Hide all tab panes
    const tabPanes = document.querySelectorAll('.tab-pane');
    tabPanes.forEach(pane => pane.classList.remove('active'));

    // Remove active class from all buttons
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => btn.classList.remove('active'));

    // Show current tab pane
    const activePane = document.getElementById(tabId);
    if (activePane) {
        activePane.classList.add('active');
    }

    // Set clicked tab button as active
    event.currentTarget.classList.add('active');
}
