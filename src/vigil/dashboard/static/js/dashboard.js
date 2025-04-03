/**
 * Main dashboard JavaScript file
 * Provides common functionality across the dashboard
 */

// Set up theme colors for consistent use with charts
const themeColors = {
    primary: '#0d6efd',
    secondary: '#6c757d',
    success: '#198754',
    danger: '#dc3545',
    warning: '#ffc107',
    info: '#0dcaf0',
    light: '#f8f9fa',
    dark: '#212529',
    chartColors: [
        '#dc3545', '#0dcaf0', '#ffc107', '#0d6efd', '#6f42c1', 
        '#fd7e14', '#20c997', '#6c757d', '#d63384', '#198754'
    ]
};

/**
 * Add animated count-up effect to number displays
 */
function animateNumbers() {
    document.querySelectorAll('.animate-number').forEach(element => {
        const target = parseInt(element.getAttribute('data-target'));
        const duration = parseInt(element.getAttribute('data-duration') || '1000');
        const increment = target / (duration / 16);
        
        let current = 0;
        const timer = setInterval(() => {
            current += increment;
            element.textContent = Math.floor(current);
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            }
        }, 16);
    });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize dashboard components
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    initTooltips();
    
    // Additional initializations
    animateNumbers();
    
    // Add filter form change handlers
    const sourceFilter = document.getElementById('source-filter');
    const tagFilter = document.getElementById('tag-filter');
    
    if (sourceFilter) {
        sourceFilter.addEventListener('change', function() {
            const form = this.closest('form');
            if (form) {
                form.submit();
            }
        });
    }
    
    if (tagFilter) {
        tagFilter.addEventListener('change', function() {
            const form = this.closest('form');
            if (form) {
                form.submit();
            }
        });
    }
});
