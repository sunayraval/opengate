document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const statusText = document.getElementById('status-text');
    const pulseDot = document.querySelector('.pulse-dot');
    const terminal = document.getElementById('terminal');
    const btnClearLogs = document.getElementById('btn-clear-logs');
    const callsList = document.getElementById('calls-list');
    const callCount = document.getElementById('call-count');

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

    async function checkStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
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
        } else {
            statusText.textContent = "Server is Offline";
            pulseDot.className = "pulse-dot stopped";
            btnStart.disabled = false;
            btnStop.disabled = true;
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
});
