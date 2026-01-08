// MeenaSetu AI Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000';
let currentImage = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkModelStatus();
    setupFileUpload();
});

// Setup file upload handlers
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    
    // Click to upload
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
        uploadArea.style.backgroundColor = 'var(--light-color)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'white';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'white';
        
        if (e.dataTransfer.files.length) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
    
    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageUpload(e.target.files[0]);
        }
    });
}

// Handle image upload
function handleImageUpload(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showAlert('Please upload an image file (JPG, PNG, etc.)', 'danger');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showAlert('File size must be less than 10MB', 'danger');
        return;
    }
    
    // Preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        currentImage = file;
        
        // Show preview
        document.getElementById('imagePreview').src = e.target.result;
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
        
        // Add log
        addLog(`Uploaded: ${file.name}`, 'upload');
    };
    reader.readAsDataURL(file);
}

// Clear uploaded image
function clearImage() {
    currentImage = null;
    document.getElementById('imagePreview').src = '';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    hideResults();
    document.getElementById('imageInput').value = '';
}

// Analyze image
async function analyzeImage() {
    if (!currentImage) {
        showAlert('Please upload an image first', 'warning');
        return;
    }
    
    // Show loading
    document.getElementById('loadingSpinner').style.display = 'block';
    hideResults();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', currentImage);
    
    try {
        // Send to API
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Hide loading
        document.getElementById('loadingSpinner').style.display = 'none';
        
        // Display results
        displayResults(result);
        
        // Add log
        addLog(`Analyzed: ${result.filename}`, 'analysis');
        
    } catch (error) {
        document.getElementById('loadingSpinner').style.display = 'none';
        showAlert('Error analyzing image: ' + error.message, 'danger');
        addLog(`Error analyzing image: ${error.message}`, 'error');
    }
}

// Display analysis results
function displayResults(result) {
    // Display species results
    if (result.species_detection && result.species_detection.status === 'success') {
        const species = result.species_detection;
        
        document.getElementById('predictedSpecies').textContent = formatSpeciesName(species.predicted_species);
        document.getElementById('speciesConfidence').textContent = 
            `Confidence: ${(species.confidence * 100).toFixed(1)}%`;
        
        // Animate confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = `${species.confidence * 100}%`;
        }, 100);
        
        // Display top predictions
        displayTopPredictions(species.top3_predictions, 'species');
        
        document.getElementById('speciesResults').style.display = 'block';
        document.getElementById('speciesResults').classList.add('fade-in');
    }
    
    // Display disease results
    if (result.disease_detection && result.disease_detection.status === 'success') {
        const disease = result.disease_detection;
        
        // Update health status
        const healthStatusDiv = document.getElementById('healthStatus');
        if (disease.is_healthy) {
            healthStatusDiv.innerHTML = `
                <span class="badge bg-success" style="font-size: 1.2em;">
                    <i class="fas fa-check-circle me-2"></i>Healthy Fish
                </span>
            `;
        } else {
            healthStatusDiv.innerHTML = `
                <span class="badge bg-danger" style="font-size: 1.2em;">
                    <i class="fas fa-exclamation-triangle me-2"></i>${formatDiseaseName(disease.predicted_disease)}
                </span>
            `;
        }
        
        document.getElementById('diseaseConfidence').textContent = 
            `Confidence: ${(disease.confidence * 100).toFixed(1)}%`;
        
        // Animate confidence bar
        const diseaseConfidenceBar = document.getElementById('diseaseConfidenceBar');
        diseaseConfidenceBar.style.width = '0%';
        setTimeout(() => {
            diseaseConfidenceBar.style.width = `${disease.confidence * 100}%`;
        }, 100);
        
        // Display disease details
        displayDiseaseDetails(disease.top3_predictions);
        
        document.getElementById('diseaseResults').style.display = 'block';
        document.getElementById('diseaseResults').classList.add('fade-in');
    }
}

// Display top predictions
function displayTopPredictions(predictions, type) {
    const container = type === 'species' ? 'topPredictions' : 'diseaseDetails';
    let html = '';
    
    predictions.forEach((pred, index) => {
        const rank = ['🥇', '🥈', '🥉'][index] || '🔸';
        const confidence = (pred.confidence * 100).toFixed(1);
        const isHealthy = pred[type]?.toLowerCase().includes('healthy') || 
                         pred.disease?.toLowerCase().includes('healthy');
        
        html += `
            <div class="prediction-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        ${rank} <strong>${type === 'species' ? pred.species : pred.disease}</strong>
                    </div>
                    <span class="badge ${isHealthy ? 'bg-success' : 'bg-primary'}">
                        ${confidence}%
                    </span>
                </div>
                <div class="progress mt-2" style="height: 5px;">
                    <div class="progress-bar ${isHealthy ? 'bg-success' : 'bg-primary'}" 
                         style="width: ${confidence}%;"></div>
                </div>
            </div>
        `;
    });
    
    document.getElementById(container).innerHTML = html;
}

// Display disease details
function displayDiseaseDetails(predictions) {
    let html = '';
    
    predictions.forEach((pred, index) => {
        const rank = ['🥇', '🥈', '🥉'][index] || '🔸';
        const confidence = (pred.confidence * 100).toFixed(1);
        const isHealthy = pred.disease.toLowerCase().includes('healthy');
        const diseaseName = formatDiseaseName(pred.disease);
        
        html += `
            <div class="prediction-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas ${isHealthy ? 'fa-check-circle text-success' : 'fa-exclamation-triangle text-danger'} me-2"></i>
                        ${rank} <strong>${diseaseName}</strong>
                    </div>
                    <span class="badge ${isHealthy ? 'bg-success' : 'bg-danger'}">
                        ${confidence}%
                    </span>
                </div>
                <div class="progress mt-2" style="height: 5px;">
                    <div class="progress-bar ${isHealthy ? 'bg-success' : 'bg-danger'}" 
                         style="width: ${confidence}%;"></div>
                </div>
            </div>
        `;
    });
    
    document.getElementById('diseaseDetails').innerHTML = html;
}

// Hide results
function hideResults() {
    document.getElementById('speciesResults').style.display = 'none';
    document.getElementById('diseaseResults').style.display = 'none';
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        const status = await response.json();
        
        // Update species model status
        const speciesElement = document.getElementById('speciesStatus');
        if (status.species_loaded) {
            speciesElement.innerHTML = `
                <div class="status-icon">
                    <i class="fas fa-fish"></i>
                </div>
                <div class="status-info">
                    <h6>Species Model</h6>
                    <span class="badge bg-success">Ready</span>
                </div>
            `;
            speciesElement.style.borderLeftColor = 'var(--secondary-color)';
        } else {
            speciesElement.innerHTML = `
                <div class="status-icon">
                    <i class="fas fa-fish"></i>
                </div>
                <div class="status-info">
                    <h6>Species Model</h6>
                    <span class="badge bg-danger">Offline</span>
                </div>
            `;
            speciesElement.style.borderLeftColor = 'var(--danger-color)';
        }
        
        // Update disease model status
        const diseaseElement = document.getElementById('diseaseStatus');
        if (status.disease_loaded) {
            diseaseElement.innerHTML = `
                <div class="status-icon">
                    <i class="fas fa-heartbeat"></i>
                </div>
                <div class="status-info">
                    <h6>Disease Model</h6>
                    <span class="badge bg-success">Ready</span>
                </div>
            `;
            diseaseElement.style.borderLeftColor = 'var(--secondary-color)';
        } else {
            diseaseElement.innerHTML = `
                <div class="status-icon">
                    <i class="fas fa-heartbeat"></i>
                </div>
                <div class="status-info">
                    <h6>Disease Model</h6>
                    <span class="badge bg-danger">Offline</span>
                </div>
            `;
            diseaseElement.style.borderLeftColor = 'var(--danger-color)';
        }
        
    } catch (error) {
        console.error('Error checking model status:', error);
        showAlert('Cannot connect to API server', 'danger');
    }
}

// Helper functions
function formatSpeciesName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatDiseaseName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.style.minWidth = '300px';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function addLog(message, type) {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // You can implement a proper log display if needed
}

function scrollToUpload() {
    document.getElementById('mainContent').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

function scrollToFeatures() {
    document.getElementById('featuresSection').scrollIntoView({ 
        behavior: 'smooth' 
    });
}