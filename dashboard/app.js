document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const statusText = document.getElementById('status-text');
    const pulseDot = document.querySelector('.pulse-dot');
    const terminal = document.getElementById('terminal');
    const btnClearLogs = document.getElementById('btn-clear-logs');
    const callsList = document.getElementById('calls-list');
    const callCount = document.getElementById('call-count');
    
    // New Elements
    const ipAddress = document.getElementById('ip-address');
    const btnLoadModel = document.getElementById('btn-load-model');
    const btnUnloadModel = document.getElementById('btn-unload-model');
    const modelSelect = document.getElementById('model-select');

    let isRunning = false;
    let ws = null;
    let callCounter = 0;

    // Initialize
    checkStatus();
    connectWebSocket();

    // Event Listeners
    btnStart.addEventListener('click', startServer);
    btnStop.addEventListener('click', stopServer);
    btnClearLogs.addEventListener('click', () => {
        terminal.innerHTML = '';
    });
    
    // Model Management Listeners
    btnLoadModel.addEventListener('click', async () => {
        btnLoadModel.disabled = true;
        try {
            const aiServerUrl = `http://${window.location.hostname}:8000/api/v1/models/load`;
            const res = await fetch(aiServerUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelSelect.value })
            });
            const data = await res.json();
            appendLog(`[Model] Load response: ${JSON.stringify(data)}`, 'log-info');
        } catch (err) {
            appendLog(`[Model] Load failed: ${err.message}`, 'log-error');
        }
        btnLoadModel.disabled = false;
    });

    btnUnloadModel.addEventListener('click', async () => {
        btnUnloadModel.disabled = true;
        try {
            const aiServerUrl = `http://${window.location.hostname}:8000/api/v1/models/unload`;
            const res = await fetch(aiServerUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelSelect.value })
            });
            const data = await res.json();
            appendLog(`[Model] Unload response: ${JSON.stringify(data)}`, 'log-info');
        } catch (err) {
            appendLog(`[Model] Unload failed: ${err.message}`, 'log-error');
        }
        btnUnloadModel.disabled = false;
    });

    async function checkStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.local_ip) {
                ipAddress.textContent = `IP: ${data.local_ip}`;
            }
            updateUI(data.running);
        } catch (err) {
            console.error("Failed to check status", err);
            updateUI(false);
        }
    }

    async function startServer() {
        btnStart.disabled = true;
        try {
            const res = await fetch('/api/start', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'started' || data.status === 'already_running') {
                updateUI(true);
            }
        } catch (err) {
            console.error("Failed to start", err);
            btnStart.disabled = false;
        }
    }

    async function stopServer() {
        btnStop.disabled = true;
        try {
            const res = await fetch('/api/stop', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'stopped' || data.status === 'already_stopped') {
                updateUI(false);
            }
        } catch (err) {
            console.error("Failed to stop", err);
            btnStop.disabled = false;
        }
    }

    function updateUI(running) {
        isRunning = running;
        if (isRunning) {
            statusText.textContent = "Server is Running";
            pulseDot.className = "pulse-dot running";
            btnStart.disabled = true;
            btnStop.disabled = false;
            btnLoadModel.disabled = false;
            btnUnloadModel.disabled = false;
        } else {
            statusText.textContent = "Server is Offline";
            pulseDot.className = "pulse-dot stopped";
            btnStart.disabled = false;
            btnStop.disabled = true;
            btnLoadModel.disabled = true;
            btnUnloadModel.disabled = true;
        }
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
        
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            appendLog("[Dashboard] Connected to Log Stream.", "log-success");
        };

        ws.onmessage = (event) => {
            const text = event.data;
            appendLog(text);
            parseLogForAPI(text);
        };

        ws.onclose = () => {
            appendLog("[Dashboard] Disconnected from Log Stream. Reconnecting in 3s...", "log-warning");
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (err) => {
            console.error("WebSocket Error", err);
        };
    }

    function appendLog(text, customClass = null) {
        const div = document.createElement('div');
        div.className = 'log-line';
        
        if (customClass) {
            div.classList.add(customClass);
        } else {
            // Basic log coloring
            if (text.includes("ERROR") || text.includes("Exception")) {
                div.classList.add("log-error");
            } else if (text.includes("WARNING")) {
                div.classList.add("log-warning");
            } else if (text.includes("INFO")) {
                div.classList.add("log-info");
            }
        }

        div.textContent = text.trim();
        terminal.appendChild(div);

        // Auto-scroll to bottom
        if (terminal.scrollHeight - terminal.scrollTop < terminal.clientHeight + 100) {
            terminal.scrollTop = terminal.scrollHeight;
        }
    }

    function parseLogForAPI(logLine) {
        // Look for typical uvicorn access logs: 
        // INFO:     127.0.0.1:54321 - "GET /health HTTP/1.1" 200 OK
        const match = logLine.match(/"(GET|POST|PUT|DELETE|OPTIONS|WS) ([^ ]+) HTTP\/1\.[01]" (\d{3})/);
        if (match) {
            const method = match[1];
            const path = match[2];
            const status = match[3];
            addCallCard(method, path, status);
            return;
        }
        
        // Also catch WebSocket connections if logged differently
        if (logLine.includes("WebSocket client connected") || logLine.includes("connection open")) {
            addCallCard("WS", "/api/v1/stream", "200");
        }
    }

    function addCallCard(method, path, status) {
        if (callCounter === 0) {
            callsList.innerHTML = '';
        }
        callCounter++;
        callCount.textContent = `${callCounter} calls`;

        const card = document.createElement('div');
        card.className = 'call-card';

        const statusClass = status.startsWith('2') ? 'status-200' : (status.startsWith('4') ? 'status-404' : 'status-500');
        const methodClass = `method-${method}`;

        const time = new Date().toLocaleTimeString();

        card.innerHTML = `
            <div>
                <span class="call-method ${methodClass}">${method}</span>
                <span class="call-path">${path}</span>
                <span class="call-status ${statusClass}">${status}</span>
            </div>
            <span class="call-time">${time}</span>
        `;

        callsList.insertBefore(card, callsList.firstChild);

        // Keep only last 50 calls
        if (callsList.children.length > 50) {
            callsList.removeChild(callsList.lastChild);
        }
    }

    // --- Tab Switching Logic ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const panels = document.querySelectorAll('.panel, .logs-section, .calls-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            navBtns.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
            
            const target = btn.getAttribute('data-target');
            
            // Hide all sections first
            document.getElementById('settings').style.display = 'none';
            document.getElementById('communicate').style.display = 'none';
            document.getElementById('dashboard').style.display = 'none';
            document.querySelector('.calls-section').style.display = 'none';

            if (target === 'dashboard') {
                document.getElementById('dashboard').style.display = 'flex';
                document.querySelector('.calls-section').style.display = 'flex';
            } else {
                document.getElementById(target).style.display = 'flex';
            }
        });
    });

    // --- Communicate Tab Logic (Push to Talk) ---
    const btnMic = document.getElementById('btn-mic');
    const chatInput = document.getElementById('chat-text-input');
    const btnSendText = document.getElementById('btn-send-text');
    const chatMessages = document.getElementById('chat-messages');

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    // Request permissions upfront if possible
    async function initAudio() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = e => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = [];
                    await sendAudioToServer(audioBlob);
                };
            } catch (err) {
                console.error("Microphone access denied", err);
                appendChatMessage("System", "Microphone access denied. Please allow microphone permissions.", "system-msg");
            }
        } else {
            appendChatMessage("System", "Audio recording is not supported in this browser.", "system-msg");
        }
    }

    // Initialize audio on first click or spacebar
    let audioInitialized = false;

    function startRecording() {
        if (isRecording) return;
        if (!mediaRecorder) {
            if (!audioInitialized) {
                audioInitialized = true;
                initAudio().then(() => {
                    if (mediaRecorder) startRecording();
                });
            }
            return;
        }
        audioChunks = [];
        mediaRecorder.start();
        isRecording = true;
        btnMic.classList.add('recording');
        appendChatMessage("System", "Recording...", "system-msg recording-indicator");
    }

    function stopRecording() {
        if (!isRecording || !mediaRecorder) return;
        mediaRecorder.stop();
        isRecording = false;
        btnMic.classList.remove('recording');
        
        // Remove recording indicator
        const indicators = document.querySelectorAll('.recording-indicator');
        indicators.forEach(el => el.remove());
        appendChatMessage("System", "Transcribing...", "system-msg transcribing-indicator");
    }

    // Event Listeners for Mic Button
    btnMic.addEventListener('mousedown', startRecording);
    btnMic.addEventListener('mouseup', stopRecording);
    btnMic.addEventListener('mouseleave', stopRecording);

    // Event Listeners for Spacebar Push-to-Talk
    document.addEventListener('keydown', (e) => {
        // Only trigger if we aren't typing in the input box and Communicate tab is active
        if (e.code === 'Space' && document.activeElement !== chatInput && document.getElementById('communicate').style.display !== 'none') {
            e.preventDefault(); // prevent scrolling
            startRecording();
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.code === 'Space' && document.getElementById('communicate').style.display !== 'none') {
            stopRecording();
        }
    });

    // Handle Audio Upload
    async function sendAudioToServer(audioBlob) {
        const formData = new FormData();
        formData.append("audio_file", audioBlob, "recording.webm");

        try {
            const aiServerUrl = `http://${window.location.hostname}:8000/api/v1/communicate/transcribe`;
            const res = await fetch(aiServerUrl, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            // Remove transcribing indicator
            const indicators = document.querySelectorAll('.transcribing-indicator');
            indicators.forEach(el => el.remove());
            
            if (data.transcription) {
                appendChatMessage('You', data.transcription, 'user-msg');
            } else if (data.detail) {
                appendChatMessage('System', `Error: ${data.detail}`, 'system-msg');
            } else {
                appendChatMessage('System', 'Transcription failed or empty (no audio detected).', 'system-msg');
            }
        } catch (err) {
            console.error("Transcription error", err);
            const indicators = document.querySelectorAll('.transcribing-indicator');
            indicators.forEach(el => el.remove());
            appendChatMessage("System", `Error transcribing: ${err.message}`, "log-error");
        }
    }

    // Handle Text Input
    async function handleSendText() {
        const text = chatInput.value.trim();
        if (!text) return;
        appendChatMessage("You", text, "user-msg");
        chatInput.value = '';
        
        try {
            const aiServerUrl = `http://${window.location.hostname}:8000/api/v1/communicate/text`;
            const res = await fetch(aiServerUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();
            
            if (data.detail) {
                appendChatMessage('System', `Error: ${data.detail}`, 'system-msg');
            }
            // Background polling will pick up the response payload from the queue.
        } catch (err) {
            console.error("Text endpoint error", err);
            appendChatMessage("System", `Error sending text: ${err.message}`, "log-error");
        }
    }

    btnSendText.addEventListener('click', handleSendText);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSendText();
    });

    function appendChatMessage(sender, text, className) {
        const div = document.createElement('div');
        div.className = `message ${className}`;
        div.innerHTML = `<strong>${sender}:</strong> ${text}`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- CONTINUOUS POLLING FOR AI RESPONSES ---
    async function pollForResponses() {
        try {
            const res = await fetch(`http://${window.location.hostname}:8000/api/v1/responses/pop`);
            if (!res.ok) return;
            const data = await res.json();
            
            if (data && data.speech) {
                // Display the speech
                appendChatMessage("AI", data.speech, "ai-msg");
                
                // Display the actions in a SEPARATE message
                let actionsHtml = `<div style="margin-top: 8px; display: flex; flex-direction: column; gap: 8px; width: 100%;">`;
                
                let actionsToDisplay = data.actions;
                if (!actionsToDisplay || actionsToDisplay.length === 0) {
                    actionsToDisplay = [{ Direction: 'None', Magnitude: '0' }];
                }
                
                actionsToDisplay.forEach((act, idx) => {
                    const dir = act.Direction || 'None';
                    const mag = act.Magnitude || '0';
                    actionsHtml += `
                        <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.2); border-left: 3px solid #4ade80; padding: 8px 12px; border-radius: 6px; font-family: monospace; font-size: 0.85em; display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #a1a1aa; font-weight: 500;">Action ${idx + 1}</span>
                            <span style="color: #4ade80; font-weight: 600; background: rgba(74, 222, 128, 0.15); padding: 2px 8px; border-radius: 4px;">${dir.toUpperCase()} ➔ ${mag}</span>
                        </div>
                    `;
                });
                actionsHtml += `</div>`;
                
                appendChatMessage("System", `<strong>Execution Plan:</strong>${actionsHtml}`, "system-msg");
                
                // Audio Playback
                if (data.audio_base64) {
                    const audioSrc = "data:audio/wav;base64," + data.audio_base64;
                    const audio = new Audio(audioSrc);
                    audio.play().catch(e => console.error("Audio play failed (maybe needs user interaction first?):", e));
                }
            }
        } catch (err) {
            // Silently fail polling (server might be down)
        }
    }
    
    // Poll every 2 seconds
    setInterval(pollForResponses, 2000);

});
