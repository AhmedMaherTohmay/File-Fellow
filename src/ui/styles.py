HEADER_HTML = """
<div class="app-header">
  <div class="app-header-icon">📜</div>
  <div class="app-header-text">
    <h1>Document Q&A Assistant</h1>
    <p>RAG pipeline · semantic memory · source citations</p>
  </div>
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #0c1220;
  --surface:    #111827;
  --panel:      #161f30;
  --border:     rgba(255,255,255,0.07);
  --border-hi:  rgba(212,160,23,0.4);
  --gold:       #d4a017;
  --gold-glow:  rgba(212,160,23,0.15);
  --gold-hi:    #f0bb3a;
  --text:       #dde4ee;
  --text-muted: #7a8fa8;
  --text-dim:   #4a5e78;
  --green:      #2ec97a;
  --red:        #e05252;
  --radius:     6px;
  --radius-lg:  10px;
  --font:       'Sora', system-ui, sans-serif;
  --mono:       'JetBrains Mono', monospace;
}

.gradio-container {
  background: var(--bg) !important;
  font-family: var(--font) !important;
  max-width: 1100px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}
.gradio-container * { box-sizing: border-box; }
body, .gradio-container, .svelte-1gfkn6j { background: var(--bg) !important; }

.app-header {
  padding: 28px 32px 20px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 14px;
}
.app-header-icon {
  width: 36px; height: 36px;
  background: var(--gold); border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
}
.app-header-text h1 { font-size: 18px; font-weight: 600; color: var(--text); margin: 0 0 2px; letter-spacing: -0.3px; }
.app-header-text p  { font-size: 12px; color: var(--text-muted); margin: 0; }

.tabs { background: transparent !important; border: none !important; }
.tab-nav {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 24px !important; gap: 0 !important;
}
.tab-nav button {
  background: transparent !important; border: none !important;
  border-bottom: 2px solid transparent !important;
  color: var(--text-muted) !important; font-family: var(--font) !important;
  font-size: 13px !important; font-weight: 500 !important;
  padding: 14px 20px !important; margin: 0 !important;
  border-radius: 0 !important; cursor: pointer;
  transition: color 0.15s, border-color 0.15s !important;
  position: relative; top: 1px;
}
.tab-nav button:hover  { color: var(--text) !important; background: transparent !important; }
.tab-nav button.selected { color: var(--gold) !important; border-bottom-color: var(--gold) !important; background: transparent !important; }

.tabitem { padding: 24px !important; background: transparent !important; }
.block, .form, .wrap, .container-fluid { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }

.section-title { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-dim); margin: 0 0 12px; }

input[type="text"], textarea, .gr-text-input, .gr-textarea {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; font-family: var(--font) !important;
  font-size: 13px !important; border-radius: var(--radius) !important;
  padding: 10px 14px !important;
  transition: border-color 0.15s, box-shadow 0.15s !important; outline: none !important;
}
input[type="text"]:focus, textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-glow) !important;
}
input::placeholder, textarea::placeholder { color: var(--text-dim) !important; }

.gradio-container label span, .gradio-container .label-wrap span {
  font-size: 11px !important; font-weight: 600 !important;
  letter-spacing: 0.8px !important; text-transform: uppercase !important;
  color: var(--text-dim) !important; font-family: var(--font) !important;
}

button.primary, .gr-button-primary, button[data-testid="primary"] {
  background: var(--gold) !important; border: none !important;
  color: #0c1220 !important; font-family: var(--font) !important;
  font-size: 13px !important; font-weight: 600 !important;
  border-radius: var(--radius) !important; padding: 10px 22px !important;
  cursor: pointer; transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
}
button.primary:hover { background: var(--gold-hi) !important; box-shadow: 0 0 20px rgba(212,160,23,0.35) !important; transform: translateY(-1px); }

button.secondary, .gr-button-secondary, button[data-testid="secondary"] {
  background: transparent !important; border: 1px solid var(--border-hi) !important;
  color: var(--gold) !important; font-family: var(--font) !important;
  font-size: 13px !important; font-weight: 500 !important;
  border-radius: var(--radius) !important; padding: 9px 18px !important;
  cursor: pointer; transition: all 0.15s !important;
}
button.secondary:hover { background: var(--gold-glow) !important; border-color: var(--gold) !important; }

button.stop, .gr-button-stop {
  background: transparent !important; border: 1px solid rgba(224,82,82,0.35) !important;
  color: var(--red) !important; font-family: var(--font) !important;
  font-size: 13px !important; border-radius: var(--radius) !important;
  padding: 9px 18px !important; cursor: pointer; transition: all 0.15s !important;
}
button.stop:hover { background: rgba(224,82,82,0.1) !important; border-color: var(--red) !important; }

.gr-dropdown, select, div[data-testid="dropdown"] > div {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; font-family: var(--font) !important;
  font-size: 13px !important; border-radius: var(--radius) !important;
}

.gr-file-upload, .file-upload, div[data-testid="file"] {
  background: var(--panel) !important; border: 2px dashed var(--border) !important;
  border-radius: var(--radius-lg) !important;
  transition: border-color 0.2s, background 0.2s !important; min-height: 120px !important;
}
.gr-file-upload:hover, div[data-testid="file"]:hover {
  border-color: var(--gold) !important; background: rgba(212,160,23,0.04) !important;
}

#chatbot {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important; height: 460px !important;
  overflow-y: auto !important; padding: 12px !important;
}
#chatbot .message { background: transparent !important; border: none !important; padding: 8px 4px !important; max-width: 100% !important; }
#chatbot .message.user { display: flex; justify-content: flex-end; }
#chatbot .message.user > div, #chatbot [data-testid="user"] > div {
  background: var(--panel) !important; border: 1px solid var(--border-hi) !important;
  border-radius: 10px 10px 2px 10px !important; padding: 10px 14px !important;
  max-width: 78% !important; color: var(--text) !important;
  font-size: 13.5px !important; line-height: 1.55 !important;
}
#chatbot .message.bot > div, #chatbot [data-testid="bot"] > div {
  background: rgba(212,160,23,0.06) !important; border: 1px solid rgba(212,160,23,0.12) !important;
  border-radius: 2px 10px 10px 10px !important; padding: 12px 16px !important;
  max-width: 88% !important; color: var(--text) !important;
  font-size: 13.5px !important; line-height: 1.6 !important;
}
#chatbot .message-wrap .avatar-container { display: none !important; }

#msg-input textarea {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important; color: var(--text) !important;
  font-family: var(--font) !important; font-size: 14px !important;
  line-height: 1.5 !important; resize: none !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
#msg-input textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 3px var(--gold-glow) !important; }

.session-pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 14px; border-radius: 20px;
  font-size: 12px; font-family: var(--font);
  color: var(--text-muted); border: 1px solid var(--border); background: var(--panel);
}
.session-pill.pill-connected { border-color: rgba(46,201,122,0.3); background: rgba(46,201,122,0.06); color: var(--green); }
.session-pill.pill-new       { border-color: var(--border-hi); background: var(--gold-glow); color: var(--gold); }
.pill-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.session-pill code { font-family: var(--mono); font-size: 11px; background: rgba(255,255,255,0.07); padding: 1px 6px; border-radius: 3px; color: inherit; }
.pill-hint { opacity: 0.65; }

#session-connect-box { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px 20px; margin-bottom: 16px; }
#session-connect-box .connect-label { font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px; }
#session-connect-box input { font-family: var(--mono) !important; font-size: 12px !important; }

#sources-panel { min-height: 0; }
.sources-wrap { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 12px; }
.sources-label { font-size: 10px; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; }
.src-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; margin-bottom: 6px; transition: border-color 0.15s; }
.src-card:hover { border-color: rgba(212,160,23,0.25); }
.src-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
.src-num { font-family: var(--mono); font-size: 11px; color: var(--gold); font-weight: 500; background: rgba(212,160,23,0.1); padding: 1px 6px; border-radius: 3px; flex-shrink: 0; }
.src-file { font-family: var(--mono); font-size: 11.5px; color: var(--text); font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.src-page { font-size: 11px; color: var(--text-muted); flex-shrink: 0; }
.src-score-wrap { width: 52px; height: 4px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; flex-shrink: 0; }
.src-score-bar { height: 100%; border-radius: 2px; transition: width 0.3s; }
.src-score-num { font-size: 10px; color: var(--text-dim); font-family: var(--mono); flex-shrink: 0; width: 28px; text-align: right; }
.src-snippet { font-size: 12px; color: var(--text-muted); line-height: 1.5; font-style: italic; }

.status-block { display: flex; flex-direction: column; gap: 6px; padding: 4px 0; }
.msg-row { display: flex; align-items: flex-start; gap: 10px; font-size: 12.5px; color: var(--text-muted); background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 12px; line-height: 1.4; }
.msg-success { border-color: rgba(46,201,122,0.25) !important; color: var(--text) !important; }
.msg-success .msg-icon { color: var(--green); }
.msg-warning { border-color: rgba(212,160,23,0.25) !important; }
.msg-warning .msg-icon { color: var(--gold); }
.msg-error   { border-color: rgba(224,82,82,0.25) !important; }
.msg-error   .msg-icon { color: var(--red); }
.msg-icon { font-size: 13px; font-weight: 700; flex-shrink: 0; margin-top: 1px; }

.doc-empty { color: var(--text-dim); font-size: 13px; text-align: center; padding: 24px; font-style: italic; }
.doc-grid { display: flex; flex-direction: column; gap: 6px; }
.doc-card { display: flex; align-items: center; gap: 12px; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 16px; transition: border-color 0.15s; }
.doc-card:hover { border-color: var(--border-hi); }
.doc-badge { font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: 0.5px; background: rgba(212,160,23,0.12); color: var(--gold); padding: 2px 7px; border-radius: 3px; flex-shrink: 0; }
.doc-name  { font-size: 13px; color: var(--text); font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-meta  { font-size: 11px; color: var(--text-dim); font-family: var(--mono); flex-shrink: 0; }

.summary-wrap { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; }
.summary-title { font-size: 12px; font-weight: 600; color: var(--gold); padding: 12px 20px 11px; border-bottom: 1px solid var(--border); font-family: var(--mono); }
.summary-body  { padding: 18px 20px; color: var(--text); font-size: 13.5px; line-height: 1.7; white-space: pre-wrap; }

.section-divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,160,23,0.35); }

footer { display: none !important; }
code { font-family: var(--mono) !important; font-size: 0.9em !important; background: rgba(255,255,255,0.07) !important; padding: 1px 5px !important; border-radius: 3px !important; }
.gr-group, .gr-box, .gr-form { background: transparent !important; border: none !important; }
"""
