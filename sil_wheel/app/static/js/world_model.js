// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Global WM angle selector state
window.wmAngleSelected = [];

function toggleWMSearchButton() {
    const objectLabel = document.getElementById("wm-class-name").value;
    const applyButton = document.getElementById("apply-wm-search-button");
    
    if (objectLabel && objectLabel !== "") {
        applyButton.disabled = false;
    } else {
        applyButton.disabled = true;
    }
    window.scrollTo(0, 0);
}

function toggleWMFilterMenu(el) {
    const adv = document.getElementById("wm-search-container");
    let activate = adv.style.display === "none";
    adv.style.display = (activate) ? "block" : "none";
    el.classList.toggle("selected", activate);

    // Check button state when showing the WM Search container
    if (adv.style.display === "block") {
        toggleWMSearchButton();
        // Initialize slider backgrounds
        const distSlider = document.getElementById("wm-max-dist");
        const timeSlider = document.getElementById("wm-min-time");
        if (distSlider) updateWMDistanceValue(distSlider.value);
        if (timeSlider) updateWMTimeValue(timeSlider.value);
    }
}

function updateWMDistanceValue(value) {
    document.getElementById("wm-max-dist-value").textContent = value;
    const slider = document.getElementById("wm-max-dist");
    const percentage = ((value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, #007bff 0%, #007bff ${percentage}%, #ddd ${percentage}%, #ddd 100%)`;
}

function updateWMTimeValue(value) {
    document.getElementById("wm-min-time-value").textContent = value;
    const slider = document.getElementById("wm-min-time");
    const percentage = ((value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, #007bff 0%, #007bff ${percentage}%, #ddd ${percentage}%, #ddd 100%)`;
}

function showWMSearchHelp() {
    let box = document.getElementById("wm-search-help-content");
    box.style.display = "block";
}

function hideWMSearchHelp() {
    let box = document.getElementById("wm-search-help-content");
    box.style.display = "none";
}

// Global function to reset WM angle selector
function resetWMAngleSelector() {
    window.wmAngleSelected = [];
    if (window.wmAngleSelector && window.wmAngleSelector.updateSelectedAngles && window.wmAngleSelector.renderSectors) {
        window.wmAngleSelector.updateSelectedAngles();
        window.wmAngleSelector.renderSectors();
    }
}

function polarToCartesian(cx, cy, r, angleDeg) {
    // SVG: 0° is 12 o'clock, increases clockwise
    const angleRad = (angleDeg - 90) * Math.PI / 180.0;
    return {
        x: cx + r * Math.cos(angleRad),
        y: cy + r * Math.sin(angleRad)
    };
}

function describeSector(cx, cy, r, startAngle, endAngle) {
    // Handle wrap-around for sectors crossing 0°
    let sweep = endAngle - startAngle;
    if (sweep <= 0) sweep += 360;
    const largeArcFlag = sweep > 180 ? "1" : "0";
    const start = polarToCartesian(cx, cy, r, startAngle);
    const end = polarToCartesian(cx, cy, r, endAngle);
    return [
        "M", cx, cy,
        "L", start.x, start.y,
        "A", r, r, 0, largeArcFlag, 1, end.x, end.y,
        "Z"
    ].join(" ");
}

function renderSectors() {
    const colors = {
        selectedFill: '#16a34a',
        unselectedFill: '#f3f4f6',
        selectedStroke: '#15803d',
        unselectedStroke: '#e5e7eb',
        hoverFill: '#f0fdf4',
        selectedText: '#f5f5f5',
        unselectedText: '#000000'
    };
    // Define sectors so that 'FRONT' is centered at 0° (right/3 o'clock)
    const sectorDefs = [
        { name: 'FRONT', label: 'Front', start: 60, end: 120 },
        { name: 'FRONT_RIGHT', label: 'Front Right', start: 120, end: 180 },
        { name: 'BACK_RIGHT', label: 'Back Right', start: 180, end: 240 },
        { name: 'BACK', label: 'Back', start: 240, end: 300 },
        { name: 'BACK_LEFT', label: 'Back Left', start: 300, end: 0 },
        { name: 'FRONT_LEFT', label: 'Front Left', start: 0, end: 60 },
    ];
    const center = 120, radius = 100, labelRadius = 70;

    const svg = document.getElementById('angle-selector-svg');
    svg.innerHTML = '';
    sectorDefs.forEach((sector, i) => {
        // Draw sector
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute('d', describeSector(center, center, radius, sector.start, sector.end));
        path.setAttribute('fill', window.wmAngleSelected.includes(sector.name) ? colors.selectedFill : colors.unselectedFill);
        path.setAttribute('stroke', window.wmAngleSelected.includes(sector.name) ? colors.selectedStroke : colors.unselectedStroke);
        path.setAttribute('stroke-width', window.wmAngleSelected.includes(sector.name) ? '2' : '1');
        path.style.cursor = 'pointer';
        path.addEventListener('mouseenter', () => {
            path.setAttribute('fill', colors.hoverFill);
        });
        path.addEventListener('mouseleave', () => {
            path.setAttribute('fill', window.wmAngleSelected.includes(sector.name) ? colors.selectedFill : colors.unselectedFill);
        });
        path.addEventListener('click', () => {
            if (window.wmAngleSelected.includes(sector.name)) {
                window.wmAngleSelected = window.wmAngleSelected.filter(s => s !== sector.name);
            } else {
                window.wmAngleSelected.push(sector.name);
            }
            updateSelectedAngles();
            renderSectors();
        });
        svg.appendChild(path);

        // Draw label
        // Place label at the center angle of the sector
        let midAngle = sector.start + ((sector.end - sector.start + 360) % 360) / 2;
        if (sector.end < sector.start) midAngle = (sector.start + ((sector.end + 360 - sector.start) / 2)) % 360;
        const labelPos = polarToCartesian(center, center, labelRadius, midAngle);
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute('x', labelPos.x);
        text.setAttribute('y', labelPos.y);
        text.setAttribute('class', 'angle-sector-label');
        text.setAttribute('fill', window.wmAngleSelected.includes(sector.name) ? colors.selectedText : colors.unselectedText);
        text.textContent = sector.label;
        svg.appendChild(text);
    });
}

function updateSelectedAngles() {
    const selectedAnglesSpan = document.getElementById('selected-angles');
    selectedAnglesSpan.textContent = window.wmAngleSelected.length ? window.wmAngleSelected.join(', ') : 'None';
    let hiddenSelect = document.getElementById('wm-angle-range');
    Array.from(hiddenSelect.options).forEach(opt => {
        opt.selected = window.wmAngleSelected.includes(opt.value);
    });
}
