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

/**
 * Metrics Viewer for Annotation Page
 * Displays simplified timeseries charts (DDC, LK, TTC) per video tile
 */

class MetricsInstance {
    constructor(clipId, container, videoElement) {
        this.clipId = clipId;
        this.container = container;
        this.videoElement = videoElement;
        this.metricsData = null;
        this.isLoading = false;
        this.hasError = false;
        this.svg = null;
        this.xScale = null;
        this.yScale = null;
        this.timeIndicator = null;
        
        // Store bound event handlers for cleanup
        this.updateTimeIndicatorHandler = null;
        
        this.initializeUI();
        this.loadMetricsData();
    }
    
    initializeUI() {
        this.container.innerHTML = '<div class="metrics-loading">Loading metrics...</div>';
    }
    
    async loadMetricsData() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        
        try {
            const response = await fetch(`/full_metrics?clip_id=${encodeURIComponent(this.clipId)}&model_name=ground_truth`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.metricsData = data;
            
            if (data && data.timestamps && data.metrics) {
                this.renderChart();
            } else {
                this.showMessage("No metrics data available");
            }
        } catch (error) {
            console.error(`Error loading metrics for ${this.clipId}:`, error);
            this.hasError = true;
            this.showMessage("Failed to load metrics");
        } finally {
            this.isLoading = false;
        }
    }
    
    showMessage(message) {
        this.container.innerHTML = `<div class="metrics-message">${message}</div>`;
    }
    
    renderChart() {
        if (!this.metricsData || !this.metricsData.timestamps || !this.metricsData.metrics) {
            this.showMessage("No metrics data available");
            return;
        }
        
        // Filter to only DDC, LK, TTC
        const metricsToShow = ["ddc", "lk", "ttc"];
        const timestamps = this.metricsData.timestamps;
        
        // Check which metrics are available
        const availableMetrics = metricsToShow.filter(m => 
            this.metricsData.metrics[m] && this.metricsData.metrics[m].length > 0
        );
        
        if (availableMetrics.length === 0) {
            this.showMessage("No DDC, LK, or TTC metrics available");
            return;
        }
        
        // Clear container
        this.container.innerHTML = '';
        
        // Get container dimensions
        const containerWidth = this.container.clientWidth || 640;
        const margin = { top: 20, right: 80, bottom: 35, left: 45 };
        const width = containerWidth - margin.left - margin.right;
        const height = 200 - margin.top - margin.bottom; // Compact height
        
        // Create SVG
        this.svg = d3.select(this.container)
            .append("svg")
            .attr("width", containerWidth)
            .attr("height", 200)
            .style("background", "#fafafa");
        
        const g = this.svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
        
        // Scales
        this.xScale = d3.scaleLinear()
            .domain([d3.min(timestamps), d3.max(timestamps)])
            .range([0, width]);
        
        // Use 0-1 range with slight padding for better visibility
        this.yScale = d3.scaleLinear()
            .domain([-0.05, 1.05])
            .range([height, 0]);
        
        // Define colors and line styles for each metric
        const metricStyles = {
            "ddc": { color: "#e74c3c", dasharray: "none", name: "DDC", tooltip: "Driving Direction Compliance" },
            "lk": { color: "#2ecc71", dasharray: "5,3", name: "LK", tooltip: "Lane Keeping" },
            "ttc": { color: "#3498db", dasharray: "2,2", name: "TTC", tooltip: "Time to Collision" }
        };
        
        // Line generator with null handling
        const line = d3.line()
            .defined(d => d != null && !isNaN(d) && isFinite(d))
            .x((d, i) => this.xScale(timestamps[i]))
            .y(d => this.yScale(d))
            .curve(d3.curveMonotoneX); // Smooth curves
        
        // Draw lines for each metric
        const linesGroup = g.append("g").attr("class", "metric-lines");
        availableMetrics.forEach(metric => {
            const values = this.metricsData.metrics[metric];
            const style = metricStyles[metric];
            
            linesGroup.append("path")
                .datum(values)
                .attr("class", `metric-line metric-${metric}`)
                .attr("fill", "none")
                .attr("stroke", style.color)
                .attr("stroke-width", 2)
                .attr("stroke-dasharray", style.dasharray)
                .attr("stroke-opacity", 0.85)
                .attr("d", line)
                // Tooltip for the metric line
                .each(function() { d3.select(this).append("title").text(style.tooltip || style.name); });
        });
        
        // Add axes
        const xAxis = d3.axisBottom(this.xScale)
            .ticks(6)
            .tickFormat(d => `${d.toFixed(1)}s`);
        
        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0,${height})`)
            .call(xAxis)
            .style("font-size", "10px");
        
        const yAxis = d3.axisLeft(this.yScale)
            .ticks(5)
            .tickFormat(d => d.toFixed(1));
        
        g.append("g")
            .attr("class", "y-axis")
            .call(yAxis)
            .style("font-size", "10px");
        
        // Compact legend in top right
        const legend = this.svg.append("g")
            .attr("class", "metrics-legend")
            .attr("transform", `translate(${width + margin.left + 10}, ${margin.top})`);
        
        availableMetrics.forEach((metric, i) => {
            const style = metricStyles[metric];
            const legendRow = legend.append("g")
                .attr("transform", `translate(0, ${i * 18})`);
            // Tooltip for the legend row
            legendRow.append("title").text(style.tooltip || style.name);
            
            // Line sample
            legendRow.append("line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("y1", 6)
                .attr("y2", 6)
                .attr("stroke", style.color)
                .attr("stroke-width", 2)
                .attr("stroke-dasharray", style.dasharray)
                .attr("stroke-opacity", 0.85);
            
            legendRow.append("text")
                .attr("x", 25)
                .attr("y", 10)
                .style("font-size", "11px")
                .style("font-weight", "500")
                .text(style.name);
        });
        
        // Add time indicator line
        this.timeIndicator = g.append("line")
            .attr("class", "time-indicator")
            .attr("y1", 0)
            .attr("y2", height)
            .attr("stroke", "#ff4444")
            .attr("stroke-width", 1.5)
            .attr("stroke-dasharray", "3,3")
            .style("pointer-events", "none")
            .style("display", "none");
        
        // Set up video sync
        this.setupVideoSync();
        
        // Add click interaction to seek video
        const clickArea = this.svg.append("rect")
            .attr("x", margin.left)
            .attr("y", margin.top)
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "transparent")
            .style("cursor", "pointer")
            .on("click", (event) => {
                if (!this.videoElement) return;
                
                const [mouseX] = d3.pointer(event, clickArea.node());
                const clickedTime = this.xScale.invert(mouseX - margin.left);
                
                if (clickedTime >= 0 && clickedTime <= this.videoElement.duration) {
                    this.videoElement.currentTime = clickedTime;
                }
            });
    }
    
    setupVideoSync() {
        if (!this.videoElement || !this.timeIndicator || !this.xScale) return;
        
        this.updateTimeIndicatorHandler = () => {
            if (this.videoElement.duration && this.videoElement.duration > 0) {
                const currentTime = this.videoElement.currentTime;
                const maxTime = d3.max(this.metricsData.timestamps);
                
                if (currentTime >= 0 && currentTime <= maxTime) {
                    this.timeIndicator
                        .style("display", "block")
                        .attr("x1", this.xScale(currentTime))
                        .attr("x2", this.xScale(currentTime));
                } else {
                    this.timeIndicator.style("display", "none");
                }
            }
        };
        
        this.videoElement.addEventListener("timeupdate", this.updateTimeIndicatorHandler);
        this.videoElement.addEventListener("loadedmetadata", this.updateTimeIndicatorHandler);
        this.videoElement.addEventListener("seeked", this.updateTimeIndicatorHandler);
        
        // Initial update
        if (this.videoElement.readyState >= 2) {
            this.updateTimeIndicatorHandler();
        }
    }
    
    destroy() {
        // Remove video event listeners
        if (this.videoElement && this.updateTimeIndicatorHandler) {
            this.videoElement.removeEventListener("timeupdate", this.updateTimeIndicatorHandler);
            this.videoElement.removeEventListener("loadedmetadata", this.updateTimeIndicatorHandler);
            this.videoElement.removeEventListener("seeked", this.updateTimeIndicatorHandler);
        }
        
        // Clean up DOM and data
        if (this.container) {
            this.container.innerHTML = '';
        }
        this.svg = null;
        this.metricsData = null;
        this.updateTimeIndicatorHandler = null;
    }
}

// Global manager for all metrics instances
class MetricsTileManager {
    constructor() {
        this.instances = new Map(); // clipId -> MetricsInstance
    }
    
    initializeMetrics(clipId, container, videoElement) {
        if (this.instances.has(clipId)) {
            return this.instances.get(clipId);
        }
        
        const instance = new MetricsInstance(clipId, container, videoElement);
        this.instances.set(clipId, instance);
        return instance;
    }
    
    destroyMetrics(clipId) {
        const instance = this.instances.get(clipId);
        if (instance) {
            instance.destroy();
            this.instances.delete(clipId);
        }
    }
    
    destroyAll() {
        this.instances.forEach(instance => instance.destroy());
        this.instances.clear();
    }
}

// Global instance
window.metricsTileManager = new MetricsTileManager();
