// Highlight Extractor - Frontend Application

const API_BASE = '';
let currentJobId = null;
let pollInterval = null;
let highlightsData = null;

// === DOM Elements ===
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const uploadBtn = document.getElementById('upload-btn');
const uploadSection = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');

// === File Handling ===
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    const validTypes = ['.wav', '.mp3', '.m4a', '.flac'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(ext)) {
        alert('Unsupported format. Please use WAV, MP3, M4A, or FLAC.');
        return;
    }
    
    if (file.size > 500 * 1024 * 1024) {
        alert('File too large. Maximum size is 500MB.');
        return;
    }
    
    // Store file on input
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    
    // Show file info
    fileInfo.classList.remove('hidden');
    fileInfo.querySelector('.file-name').textContent = file.name;
    fileInfo.querySelector('.file-size').textContent = formatSize(file.size);
    dropZone.classList.add('hidden');
    uploadBtn.disabled = false;
}

function removeFile() {
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    dropZone.classList.remove('hidden');
    uploadBtn.disabled = true;
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// === Upload ===
async function uploadFile() {
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_n', document.getElementById('top_n').value);
    
    const preset = document.getElementById('preset').value;
    if (preset !== 'default') {
        formData.append('keyword_preset', preset);
    }
    
    const speakers = document.getElementById('speakers').value;
    if (speakers) {
        formData.append('expected_num_speakers', speakers);
    }
    
    // Show progress
    uploadSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    updateProgress('QUEUED', 0);
    
    try {
        const response = await fetch(`${API_BASE}/v1/jobs`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail?.message || 'Upload failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Start polling
        startPolling();
        
    } catch (error) {
        showError(error.message);
    }
}

// === Polling ===
function startPolling() {
    pollInterval = setInterval(pollJob, 2000);
    pollJob();
}

async function pollJob() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`${API_BASE}/v1/jobs/${currentJobId}`);
        const data = await response.json();
        
        updateProgress(data.status, getProgressPercent(data.status));
        
        if (data.status === 'DONE') {
            clearInterval(pollInterval);
            await loadHighlights();
        } else if (data.status === 'FAILED') {
            clearInterval(pollInterval);
            showError(data.error?.message || 'Job failed');
        }
        
    } catch (error) {
        console.error('Poll error:', error);
    }
}

function getProgressPercent(status) {
    const stages = ['QUEUED', 'INGESTING', 'TRANSCRIBING', 'DIARIZING', 'ALIGNING', 'EXTRACTING_FEATURES', 'SCORING', 'DONE'];
    const index = stages.indexOf(status);
    return Math.round((index / (stages.length - 1)) * 100);
}

function updateProgress(status, percent) {
    document.getElementById('progress-fill').style.width = `${percent}%`;
    document.getElementById('progress-status').textContent = status.replace(/_/g, ' ');
    
    // Update stage list
    const stages = document.querySelectorAll('.stage');
    const statusOrder = ['QUEUED', 'INGESTING', 'TRANSCRIBING', 'DIARIZING', 'ALIGNING', 'EXTRACTING_FEATURES', 'SCORING', 'DONE'];
    const currentIndex = statusOrder.indexOf(status);
    
    stages.forEach((stage, index) => {
        stage.classList.remove('active', 'completed');
        if (index < currentIndex) {
            stage.classList.add('completed');
        } else if (index === currentIndex) {
            stage.classList.add('active');
        }
    });
}

// === Results ===
async function loadHighlights() {
    try {
        const response = await fetch(`${API_BASE}/v1/jobs/${currentJobId}/highlights`);
        const data = await response.json();
        
        highlightsData = data;
        
        // Show results
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Meta info
        document.getElementById('audio-duration').textContent = `⏱️ ${formatDuration(data.audio_duration_s)}`;
        document.getElementById('num-speakers').textContent = `👥 ${data.num_speakers_detected} speakers`;
        
        // Quality warning
        const warningEl = document.getElementById('quality-warning');
        if (data.quality_warning) {
            warningEl.textContent = `⚠️ ${data.quality_warning}`;
            warningEl.classList.remove('hidden');
        } else {
            warningEl.classList.add('hidden');
        }
        
        // Render highlights
        const listEl = document.getElementById('highlights-list');
        listEl.innerHTML = data.highlights.map((h, i) => `
            <div class="highlight-item ${h.low_confidence ? 'low-confidence' : ''}">
                <div class="highlight-header">
                    <span class="highlight-rank">#${i + 1}</span>
                    <span class="highlight-score">Score: ${h.score.toFixed(3)}</span>
                </div>
                <div class="highlight-time">
                    🕐 ${formatDuration(h.start_s)} → ${formatDuration(h.end_s)} 
                    (${formatDuration(h.end_s - h.start_s)}) · ${h.speaker}
                </div>
                <div class="highlight-text">"${h.transcript_excerpt}"</div>
                ${h.reasons.length > 0 ? `
                    <div class="highlight-reasons">
                        ${h.reasons.map(r => `<span class="reason-tag">${r}</span>`).join('')}
                    </div>
                ` : ''}
                ${h.low_confidence ? '<div class="reason-tag" style="background: rgba(245,158,11,0.2); color: #f59e0b;">⚠️ Low confidence</div>' : ''}
            </div>
        `).join('');
        
    } catch (error) {
        showError('Failed to load highlights: ' + error.message);
    }
}

// === Download ===
function downloadResults() {
    if (!highlightsData) return;
    
    const blob = new Blob([JSON.stringify(highlightsData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `highlights-${currentJobId}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// === Reset ===
function resetApp() {
    currentJobId = null;
    highlightsData = null;
    
    uploadSection.classList.remove('hidden');
    progressSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    
    removeFile();
}

// === Error ===
function showError(message) {
    uploadSection.classList.add('hidden');
    progressSection.classList.add('hidden');
    errorSection.classList.remove('hidden');
    document.getElementById('error-message').textContent = message;
}
