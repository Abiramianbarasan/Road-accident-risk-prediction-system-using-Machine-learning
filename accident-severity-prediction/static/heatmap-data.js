// heatmap-data.js

document.addEventListener("DOMContentLoaded", () => {
    const map = L.map('map').setView([20.5937, 78.9629], 5); // Centered on India

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    fetch('/heatmap-data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error fetching heatmap data: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (!Array.isArray(data) || data.length === 0) {
                document.getElementById('map').innerHTML = '<div style="text-align: center; padding: 20px;">No data available for visualization</div>';
                return;
            }

            const heatData = data.map(p => [p.location.lat, p.location.lng, p.weight]);

            L.heatLayer(heatData, {
                radius: 25,
                blur: 15,
                maxZoom: 10,
                max: 1.0,
                gradient: {
                    0.4: 'blue',
                    0.6: 'lime',
                    0.8: 'orange',
                    1.0: 'red'
                }
            }).addTo(map);

            const bounds = L.latLngBounds(heatData.map(p => [p[0], p[1]]));
            map.fitBounds(bounds);
        })
        .catch(error => {
            console.error("Heatmap load error:", error);
            document.getElementById('map').innerHTML = `<div style="color: red; padding: 20px;">Error loading heatmap: ${error.message}</div>`;
        });
});
