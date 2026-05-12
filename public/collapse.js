// Collapse "Raw code" blocks + Flowchart injection + Monitor
(function() {
    // ── Code block collapse ──
    function initCollapse() {
        document.querySelectorAll('code.whitespace-pre-wrap').forEach(code => {
            if (code.textContent.length < 300) return;
            const outer = code.closest('.rounded-lg');
            if (!outer || outer.dataset.cbInit) return;
            outer.dataset.cbInit = '1';
            outer.classList.add('cb-collapsed');
            outer.style.cursor = 'pointer';
            outer.addEventListener('click', function(e) {
                if (e.target.closest('button')) return;
                outer.classList.toggle('cb-collapsed');
                outer.classList.toggle('cb-expanded');
            });
        });
    }
    initCollapse();
    setInterval(initCollapse, 2000);

    // ── Flowchart injection ──
    function injectFlowchart() {
        const aside = document.querySelector('aside');
        if (!aside || aside.dataset.fcInit) return;
        const fcLink = document.querySelector('[href*="flowchart"]');
        if (!fcLink) return;
        aside.dataset.fcInit = '1';
        const panel = document.createElement('div');
        panel.style.cssText = 'padding:8px;font-size:11px;border-top:1px solid #333;max-height:40vh;overflow:auto;';
        panel.innerHTML = '<b>Flowchart</b><br><small>Click to open</small>';
        panel.addEventListener('click', () => fcLink.click());
        aside.appendChild(panel);
    }
    setInterval(injectFlowchart, 3000);

    // ── HHSDD Monitor: 固定在 disclaimer 旁边的状态栏 ──
    //     使用 MutationObserver 防止 React 重渲染时丢失
    (function() {
        var statusEl = null;
        var currentText = '';

        function createStatus(parent) {
            if (statusEl && document.contains(statusEl)) return;
            statusEl = document.createElement('span');
            statusEl.id = 'hhsdd-monitor-status';
            statusEl.style.cssText = 'margin-left:12px;font:11px/1.4 monospace;color:#888;vertical-align:middle;white-space:nowrap;';
            statusEl.textContent = currentText;
            parent.appendChild(statusEl);
        }

        // MutationObserver: 监控 disclaimer 出现/重新渲染
        var observer = new MutationObserver(function() {
            var divs = document.querySelectorAll('div');
            for (var i = 0; i < divs.length; i++) {
                var t = divs[i].textContent;
                var h = divs[i].offsetHeight;
                if (t.includes('大语言模型') && t.length < 100 && h > 0 && h <= 20
                    && !divs[i].querySelector('#hhsdd-monitor-status')) {
                    createStatus(divs[i]);
                    break;
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        // 初始检查
        observer.takeRecords();

        // 轮询 monitor.json
        var toolList = [];
        var _lastKey = '';
        setInterval(function() {
            fetch('/public/monitor.json').then(function(r) {
                if (!r.ok) return null;
                return r.json();
            }).then(function(d) {
                if (!d) return;
                var key = JSON.stringify(d);
                if (key === '{}' || key === _lastKey) return;
                _lastKey = key;
                var parts = [];
                if (d.cache_pct != null) parts.push('Cache ' + d.cache_pct + '%');
                if (d.energy != null) parts.push('⚡' + Math.round(d.energy).toLocaleString());
                if (d.ctx_pct != null) parts.push('Ctx ' + d.ctx_pct + '%');
                if (d.step != null) parts.push('S' + d.step);
                if (d.tokens != null) parts.push(d.tokens.toLocaleString() + 'tok');
                if (d.tools && d.tools.length) parts.push('⚒' + d.tools.length);
                currentText = parts.join(' · ');
                if (statusEl && document.contains(statusEl)) {
                    statusEl.textContent = currentText;
                }
            }).catch(function() {});
        }, 500);

        window.__hhsdd_reset_monitor = function() { toolList = []; currentText = ''; };
        console.log('[HHSDD] Monitor loaded');
    })();
})();
