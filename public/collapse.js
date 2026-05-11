// Collapse "Raw code" blocks in sidebar tool output
(function() {
    function initCollapse() {
        document.querySelectorAll('code.whitespace-pre-wrap').forEach(code => {
            if (code.textContent.length < 300) return;
            // Find the outer "Raw code" container
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
})();
