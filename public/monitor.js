/**
 * HHSDD 资源监控面板 — 固定在页面底部的状态栏
 * 通过 window.__hhsdd_update(data) 接收更新
 */
(function() {
  'use strict';

  // 创建面板 DOM
  const bar = document.createElement('div');
  bar.id = 'hhsdd-monitor';
  bar.innerHTML = `
    <style>
      #hhsdd-monitor {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        height: 32px;
        background: rgba(30, 30, 30, 0.92);
        color: #e0e0e0;
        font: 12px/32px "Cascadia Code", "Fira Code", "Consolas", monospace;
        display: flex;
        align-items: center;
        padding: 0 16px;
        gap: 20px;
        z-index: 9999;
        border-top: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(8px);
        user-select: none;
        transition: opacity 0.3s;
      }
      #hhsdd-monitor.hidden { opacity: 0; pointer-events: none; }
      #hhsdd-monitor .m-item {
        display: flex; align-items: center; gap: 4px;
        white-space: nowrap;
      }
      #hhsdd-monitor .m-label {
        color: #888; font-size: 11px;
      }
      #hhsdd-monitor .m-value {
        font-weight: 600;
      }
      #hhsdd-monitor .m-bar {
        display: inline-block;
        width: 60px; height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        overflow: hidden;
        vertical-align: middle;
      }
      #hhsdd-monitor .m-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s;
      }
      .m-cache .m-bar-fill { background: #4caf50; }
      .m-energy .m-bar-fill { background: #2196f3; }
      .m-ctx .m-bar-fill { background: #ff9800; }
      .m-sep { color: #444; }
    </style>
    <div class="m-item m-cache">
      <span class="m-label">Cache</span>
      <span class="m-value" id="m-cache-val">--</span>
      <span class="m-bar"><span class="m-bar-fill" id="m-cache-bar" style="width:0%"></span></span>
    </div>
    <span class="m-sep">│</span>
    <div class="m-item m-energy">
      <span class="m-label">Energy</span>
      <span class="m-value" id="m-energy-val">--</span>
      <span class="m-bar"><span class="m-bar-fill" id="m-energy-bar" style="width:0%"></span></span>
    </div>
    <span class="m-sep">│</span>
    <div class="m-item m-ctx">
      <span class="m-label">Context</span>
      <span class="m-value" id="m-ctx-val">--</span>
      <span class="m-bar"><span class="m-bar-fill" id="m-ctx-bar" style="width:0%"></span></span>
    </div>
    <span class="m-sep">│</span>
    <div class="m-item">
      <span class="m-label">Step</span>
      <span class="m-value" id="m-step-val">--</span>
    </div>
    <span class="m-sep">│</span>
    <div class="m-item">
      <span class="m-label">Tokens</span>
      <span class="m-value" id="m-tokens-val">--</span>
    </div>
    <span class="m-sep">│</span>
    <div class="m-item">
      <span class="m-label">Tools</span>
      <span class="m-value" id="m-tools-val">--</span>
    </div>
  `;
  document.body.appendChild(bar);

  // 工具计数器
  let toolNames = [];

  // 全局更新接口
  window.__hhsdd_update = function(data) {
    const el = (id) => document.getElementById(id);

    if (data.cache_pct !== undefined) {
      const v = el('m-cache-val');
      const b = el('m-cache-bar');
      if (v) v.textContent = data.cache_pct + '%';
      if (b) b.style.width = data.cache_pct + '%';
    }

    if (data.energy !== undefined) {
      const v = el('m-energy-val');
      const b = el('m-energy-bar');
      const pct = data.energy_pct || 0;
      if (v) v.textContent = Math.round(data.energy).toLocaleString();
      if (b) b.style.width = Math.min(100, pct) + '%';
    }

    if (data.ctx_pct !== undefined) {
      const v = el('m-ctx-val');
      const b = el('m-ctx-bar');
      if (v) v.textContent = Math.round(data.ctx_pct) + '%';
      if (b) b.style.width = Math.min(100, data.ctx_pct) + '%';
    }

    if (data.step !== undefined) {
      const v = el('m-step-val');
      if (v) v.textContent = data.step;
    }

    if (data.tokens !== undefined) {
      const v = el('m-tokens-val');
      if (v) v.textContent = data.tokens.toLocaleString();
    }

    if (data.tool) {
      toolNames.push(data.tool);
      const v = el('m-tools-val');
      if (v) v.textContent = toolNames.length + ' (' + toolNames.slice(-3).join(', ') + ')';
    }

    if (data.reset) {
      toolNames = [];
      ['m-cache-val','m-energy-val','m-ctx-val','m-step-val','m-tokens-val','m-tools-val']
        .forEach(id => { const e = el(id); if(e) e.textContent = '--'; });
      ['m-cache-bar','m-energy-bar','m-ctx-bar']
        .forEach(id => { const e = el(id); if(e) e.style.width = '0%'; });
    }

    // 显示面板
    bar.classList.remove('hidden');
  };

  // 隐藏面板（闲置时）
  window.__hhsdd_hide = function() {
    bar.classList.add('hidden');
  };

  // 初始隐藏
  bar.classList.add('hidden');

  console.log('[HHSDD] Monitor panel loaded');
})();
