document.querySelector('form').addEventListener('submit', function(e) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = 'flex';

    // Simulate loading (remove in actual implementation)
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
        // Uncomment next line in actual implementation
        // this.submit();
    }, 3000);
}); 