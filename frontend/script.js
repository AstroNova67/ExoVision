// Global variables
let currentModel = 'koi';
const API_BASE_URL = 'http://localhost:8000';

// Model configurations
const modelConfigs = {
    koi: {
        name: 'KOI (Kepler)',
        icon: 'fas fa-star',
        endpoint: '/predict/koi',
        parameters: [
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
            'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_srad'
        ]
    },
    tess: {
        name: 'TESS',
        icon: 'fas fa-satellite',
        endpoint: '/predict/tess',
        parameters: [
            'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 'pl_insol',
            'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg'
        ]
    }
};

// DOM elements
const modelButtons = document.querySelectorAll('.model-btn');
const paramGroups = document.querySelectorAll('.param-group');
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
const predictionsDiv = document.getElementById('predictions');
const predictBtn = document.querySelector('.predict-btn');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeModelSelector();
    initializeForm();
    setupEventListeners();
});

// Model selector functionality
function initializeModelSelector() {
    modelButtons.forEach(button => {
        button.addEventListener('click', function() {
            const model = this.getAttribute('data-model');
            switchModel(model);
        });
    });
}

function switchModel(model) {
    // Update active button
    modelButtons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-model="${model}"]`).classList.add('active');
    
    // Update active parameter group
    paramGroups.forEach(group => group.classList.remove('active'));
    document.getElementById(`${model}-params`).classList.add('active');
    
    // Update current model
    currentModel = model;
    
    // Clear previous results
    hideResults();
    
    // Reset form
    form.reset();
}

// Form functionality
function initializeForm() {
    // Add input validation
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', validateInput);
        input.addEventListener('blur', validateInput);
    });
}

function validateInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);
    
    // Remove previous validation classes
    input.classList.remove('invalid', 'valid');
    
    if (input.value === '') {
        return; // Allow empty for optional fields
    }
    
    if (isNaN(value) || value < 0) {
        input.classList.add('invalid');
        showFieldError(input, 'Please enter a valid positive number');
    } else {
        input.classList.add('valid');
        hideFieldError(input);
    }
}

function showFieldError(input, message) {
    hideFieldError(input);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'field-error';
    errorDiv.textContent = message;
    errorDiv.style.color = '#e53e3e';
    errorDiv.style.fontSize = '0.8rem';
    errorDiv.style.marginTop = '4px';
    
    input.parentNode.appendChild(errorDiv);
}

function hideFieldError(input) {
    const existingError = input.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
}

// Event listeners
function setupEventListeners() {
    form.addEventListener('submit', handleFormSubmit);
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            handleFormSubmit(event);
        }
    });
}

// Form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    if (!validateForm()) {
        showNotification('Please fill in all required fields with valid values', 'error');
        return;
    }
    
    const formData = collectFormData();
    await makePrediction(formData);
}

function validateForm() {
    const config = modelConfigs[currentModel];
    let isValid = true;
    
    config.parameters.forEach(param => {
        const input = document.getElementById(param);
        if (input) {
            const value = input.value.trim();
            
            // Required field validation
            if (value === '') {
                input.classList.add('invalid');
                showFieldError(input, 'This field is required');
                isValid = false;
            } else if (isNaN(parseFloat(value)) || parseFloat(value) < 0) {
                input.classList.add('invalid');
                showFieldError(input, 'Please enter a valid positive number');
                isValid = false;
            } else {
                input.classList.remove('invalid');
                hideFieldError(input);
            }
        }
    });
    
    return isValid;
}

function collectFormData() {
    const config = modelConfigs[currentModel];
    const data = {};
    
    config.parameters.forEach(param => {
        const input = document.getElementById(param);
        if (input) {
            data[param] = parseFloat(input.value) || 0;
        }
    });
    
    return data;
}

// API communication
async function makePrediction(data) {
    showLoading();
    
    try {
        const config = modelConfigs[currentModel];
        console.log('Making request to:', `${API_BASE_URL}${config.endpoint}`);
        console.log('Request data:', data);
        
        const response = await fetch(`${API_BASE_URL}${config.endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Response error:', errorText);
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        
        const results = await response.json();
        console.log('Results received:', results);
        displayResults(results);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`Failed to get prediction: ${error.message}`, 'error');
        hideLoading();
    }
}

// Results display
function showLoading() {
    resultsSection.classList.remove('hidden');
    loadingDiv.style.display = 'block';
    predictionsDiv.innerHTML = '';
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
}

function hideLoading() {
    loadingDiv.style.display = 'none';
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<i class="fas fa-magic"></i> Predict Classification';
}

function displayResults(results) {
    hideLoading();
    
    predictionsDiv.innerHTML = '';
    
    // Create prediction cards for each model
    const modelNames = {
        'adaboost': 'AdaBoost',
        'stacking': 'Stacking',
        'forest_classifier': 'Random Forest',
        'subspace': 'Subspace',
        'extra_trees': 'Extra Trees'
    };
    
    const classificationLabels = {
        0: 'CONFIRMED',
        1: 'CANDIDATE', 
        2: 'FALSE POSITIVE'
    };
    
    Object.entries(results.predictions).forEach(([model, prediction], index) => {
        const card = createPredictionCard(
            modelNames[model] || model,
            prediction.classification,
            prediction.confidence,
            classificationLabels[prediction.classification] || 'UNKNOWN'
        );
        
        // Add animation delay
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
        
        predictionsDiv.appendChild(card);
    });
    
    // Add summary
    const summary = createSummaryCard(results);
    predictionsDiv.appendChild(summary);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createPredictionCard(modelName, classification, confidence, label) {
    const card = document.createElement('div');
    card.className = 'prediction-card';
    
    const classificationClass = label.toLowerCase().replace(' ', '-');
    
    card.innerHTML = `
        <h3>
            <span class="model-name">${modelName}</span>
            <i class="fas fa-brain"></i>
        </h3>
        <div class="prediction">
            <span class="classification ${classificationClass}">${label}</span>
        </div>
        <div class="confidence">
            Confidence: ${(confidence * 100).toFixed(1)}%
        </div>
    `;
    
    return card;
}

function createSummaryCard(results) {
    const card = document.createElement('div');
    card.className = 'prediction-card';
    card.style.borderLeftColor = '#48bb78';
    
    // Find most common prediction
    const predictions = Object.values(results.predictions);
    const classifications = predictions.map(p => p.classification);
    const mostCommon = mode(classifications);
    
    const classificationLabels = {
        0: 'CONFIRMED',
        1: 'CANDIDATE', 
        2: 'FALSE POSITIVE'
    };
    
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
    
    card.innerHTML = `
        <h3>
            <span class="model-name" style="background: #48bb78;">SUMMARY</span>
            <i class="fas fa-chart-line"></i>
        </h3>
        <div class="prediction">
            <span class="classification ${classificationLabels[mostCommon].toLowerCase().replace(' ', '-')}">
                ${classificationLabels[mostCommon]}
            </span>
        </div>
        <div class="confidence">
            Average Confidence: ${(avgConfidence * 100).toFixed(1)}%
        </div>
        <div style="margin-top: 15px; font-size: 0.9rem; color: #4a5568;">
            Based on ${predictions.length} different machine learning models
        </div>
    `;
    
    return card;
}

// Utility functions
function mode(arr) {
    return arr.sort((a, b) =>
        arr.filter(v => v === a).length - arr.filter(v => v === b).length
    ).pop();
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#fed7d7' : '#bee3f8'};
        color: ${type === 'error' ? '#742a2a' : '#2a4365'};
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    .invalid {
        border-color: #e53e3e !important;
        box-shadow: 0 0 0 3px rgba(229, 62, 62, 0.1) !important;
    }
    
    .valid {
        border-color: #48bb78 !important;
        box-shadow: 0 0 0 3px rgba(72, 187, 120, 0.1) !important;
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
