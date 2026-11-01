// Language switcher for documentation
document.addEventListener('DOMContentLoaded', function() {
    // Create language switcher
    var switcher = document.createElement('div');
    switcher.className = 'language-switcher';
    
    // Get current path
    var path = window.location.pathname;
    var isRussian = path.includes('/ru/');
    
    // Build URLs
    var enUrl, ruUrl;
    if (isRussian) {
        enUrl = path.replace('/ru/', '/en/');
        ruUrl = path;
    } else if (path.includes('/en/')) {
        enUrl = path;
        ruUrl = path.replace('/en/', '/ru/');
    } else {
        // Default paths
        enUrl = path;
        ruUrl = path.replace(/^\//, '/ru/');
    }
    
    // Create links
    var enLink = document.createElement('a');
    enLink.href = enUrl;
    enLink.textContent = 'EN';
    if (!isRussian) enLink.className = 'active';
    
    var ruLink = document.createElement('a');
    ruLink.href = ruUrl;
    ruLink.textContent = 'RU';
    if (isRussian) ruLink.className = 'active';
    
    switcher.appendChild(enLink);
    switcher.appendChild(ruLink);
    document.body.appendChild(switcher);
});
