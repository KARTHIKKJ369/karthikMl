document.addEventListener('DOMContentLoaded', () => {
  const htmlRoot = document.documentElement;
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const chatViewport = document.getElementById('chatViewport');
  const heroContainer = document.getElementById('heroContainer');
  const messageFeed = document.getElementById('messageFeed');
  const sidebar = document.getElementById('sidebar');
  const sidebarToggle = document.getElementById('sidebarToggle');
  const newChatBtn = document.getElementById('newChatBtn');

  const metricsPill = document.getElementById('metricsPill');
  const metricSpeed = document.getElementById('metricSpeed');
  const metricTime = document.getElementById('metricTime');

  const themeButtons = document.querySelectorAll('.theme-btn');
  const chipButtons = document.querySelectorAll('.chip-item, .kb-stats-card');

  // 1. Theme Management (Void, Stone, Rust)
  const savedTheme = localStorage.getItem('ridge_theme') || 'void';
  setTheme(savedTheme);

  themeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const theme = btn.dataset.theme;
      setTheme(theme);
    });
  });

  function setTheme(theme) {
    htmlRoot.setAttribute('data-theme', theme);
    localStorage.setItem('ridge_theme', theme);
    themeButtons.forEach((b) => {
      b.classList.toggle('active', b.dataset.theme === theme);
    });
  }

  // 2. Sidebar Toggle
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  // 3. New Chat Button (Reset view)
  newChatBtn.addEventListener('click', () => {
    messageFeed.innerHTML = '';
    heroContainer.style.display = 'flex';
    metricsPill.style.display = 'none';
    chatInput.value = '';
    chatInput.focus();
  });

  // 4. Quick Prompt Chips & Knowledge Cards
  chipButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const prompt = btn.dataset.prompt;
      if (prompt) {
        chatInput.value = prompt;
        handleSubmit();
      }
    });
  });

  // 5. Auto-growing Textarea & Enter key listener
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  });

  sendBtn.addEventListener('click', handleSubmit);

  // 6. Message Submission & Generation
  async function handleSubmit() {
    const question = chatInput.value.trim();
    if (!question) return;

    // Hide hero if visible
    heroContainer.style.display = 'none';

    // Render User Message
    appendMessage('user', question);

    // Clear input & reset height
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Render Assistant Placeholder
    const assistantBubble = appendMessage('assistant', '', true);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: question,
          temperature: 0.2,
          max_new_tokens: 250,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Update Assistant Bubble
      assistantBubble.textContent = data.answer || data.text || 'No response returned.';

      // Update Telemetry in Navbar
      metricSpeed.textContent = `${data.tokens_per_sec.toFixed(1)} tok/s`;
      metricTime.textContent = `${Math.round(data.time_ms)} ms`;
      metricsPill.style.display = 'flex';
    } catch (err) {
      assistantBubble.innerHTML = `<span style="color: #ef4444;">⚠️ Error: ${err.message}</span>`;
    } finally {
      sendBtn.disabled = false;
      chatViewport.scrollTo({ top: chatViewport.scrollHeight, behavior: 'smooth' });
    }
  }

  function appendMessage(role, text, isStreaming = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message';

    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';

    if (role === 'user') {
      metaDiv.innerHTML = '<span class="user-meta-tag">You</span>';
    } else {
      metaDiv.innerHTML = '<span class="assistant-meta-tag">Karthik Jayan</span><span>·</span><span>MLX 1.5B</span>';
    }

    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${role}-bubble`;
    bubble.textContent = text;

    if (isStreaming) {
      const dot = document.createElement('span');
      dot.className = 'streaming-dot';
      bubble.appendChild(dot);
    }

    msgDiv.appendChild(metaDiv);
    msgDiv.appendChild(bubble);
    messageFeed.appendChild(msgDiv);

    chatViewport.scrollTo({ top: chatViewport.scrollHeight, behavior: 'smooth' });
    return bubble;
  }
});
