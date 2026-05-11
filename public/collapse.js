// Collapse "Raw code" blocks + Flowchart injection
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

    // ── Flowchart injection into left aside ──
    function injectFlowchart() {
        const aside = document.querySelector('aside');
        if (!aside || aside.dataset.fcInit) return;
        // Try to read flowchart from the page content
        const fcLink = document.querySelector('[href*="flowchart"]');
        if (!fcLink) return;
        aside.dataset.fcInit = '1';
        // Create mini flowchart panel
        const panel = document.createElement('div');
        panel.style.cssText = 'padding:8px;font-size:11px;border-top:1px solid #333;max-height:40vh;overflow:auto;';
        panel.innerHTML = '<b>Flowchart</b><br><small>Click to open in full page</small>';
        panel.addEventListener('click', () => fcLink.click());
        aside.appendChild(panel);
    }
    setInterval(injectFlowchart, 3000);
})();
