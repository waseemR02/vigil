/**
 * Charts for the Vigil Cybersecurity Dashboard
 * Uses Chart.js for visualizations
 */

// Common chart configuration options
const chartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                color: '#f8f9fa'
            }
        },
        tooltip: {
            backgroundColor: 'rgba(33, 37, 41, 0.9)',
            titleColor: '#f8f9fa',
            bodyColor: '#f8f9fa',
            borderColor: '#444',
            borderWidth: 1,
            padding: 10
        }
    },
    scales: {
        x: {
            ticks: {
                color: '#adb5bd'
            },
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            }
        },
        y: {
            beginAtZero: true,
            ticks: {
                color: '#adb5bd'
            },
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            }
        }
    }
};

/**
 * Initialize incidents by source bar chart
 * @param {string} canvasId - Canvas element ID
 * @param {Array} sources - List of sources
 * @param {Array} incidents - List of incidents
 */
function initIncidentsBySourceChart(canvasId, sources, incidents) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !sources || !incidents || !Array.isArray(sources) || !Array.isArray(incidents)) return;
    
    // Count incidents by source
    const sourceCounts = {};
    sources.forEach(source => {
        if (source && source.id) {
            sourceCounts[source.id] = 0;
        }
    });
    
    incidents.forEach(incident => {
        if (incident && incident.source && incident.source.id) {
            sourceCounts[incident.source.id] = (sourceCounts[incident.source.id] || 0) + 1;
        }
    });
    
    // Prepare data for chart
    const sourceNames = sources
        .filter(source => source && source.name)
        .map(source => source.name);
    
    const counts = sources
        .filter(source => source && source.id)
        .map(source => sourceCounts[source.id] || 0);
    
    // Don't create chart if we have no data
    if (sourceNames.length === 0) {
        canvas.parentNode.innerHTML = '<div class="text-center text-muted py-5">No source data available for chart</div>';
        return;
    }
    
    // Create chart
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: sourceNames,
            datasets: [{
                label: 'Incidents',
                data: counts,
                backgroundColor: 'rgba(220, 53, 69, 0.6)',
                borderColor: 'rgba(220, 53, 69, 1)',
                borderWidth: 1
            }]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: {
                    display: false,
                    text: 'Incidents by Source',
                    color: '#f8f9fa'
                }
            }
        }
    });
}

/**
 * Initialize incidents timeline chart
 * @param {string} canvasId - Canvas element ID
 * @param {Array} incidents - List of incidents
 */
function initIncidentsTimelineChart(canvasId, incidents) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !incidents || !Array.isArray(incidents) || incidents.length === 0) {
        if (canvas) {
            canvas.parentNode.innerHTML = '<div class="text-center text-muted py-5">No incident data available for timeline</div>';
        }
        return;
    }
    
    // Group incidents by date (last 30 days)
    const today = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    // Create an array of dates for the last 30 days
    const dates = [];
    const labels = [];
    for (let i = 0; i < 30; i++) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        date.setHours(0, 0, 0, 0);
        
        const formattedDate = date.toISOString().split('T')[0];
        dates.unshift(formattedDate);
        
        // Format label as "MMM DD"
        const label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        labels.unshift(label);
    }
    
    // Count incidents per date
    const incidentCounts = Array(30).fill(0);
    
    incidents.forEach(incident => {
        if (!incident || !incident.created_at) return;
        
        try {
            let incidentDate;
            if (typeof incident.created_at === 'string') {
                // Handle ISO date strings
                incidentDate = incident.created_at.split('T')[0];
            } else if (incident.created_at instanceof Date) {
                incidentDate = incident.created_at.toISOString().split('T')[0];
            } else {
                return; // Skip if we can't parse the date
            }
            
            const index = dates.indexOf(incidentDate);
            if (index !== -1) {
                incidentCounts[index]++;
            }
        } catch (e) {
            console.error('Error processing incident date:', e);
        }
    });
    
    // Create chart
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Incidents',
                data: incidentCounts,
                backgroundColor: 'rgba(13, 202, 240, 0.2)',
                borderColor: 'rgba(13, 202, 240, 1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: {
                    display: false,
                    text: 'Incidents Timeline (Last 30 Days)',
                    color: '#f8f9fa'
                }
            }
        }
    });
}
