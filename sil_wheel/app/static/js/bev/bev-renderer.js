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
 * Canvas renderer for driving scene visualization.
 * Handles drawing of vehicles, road boundaries, lane lines, and other elements.
 */

class SceneRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.transform = new CoordinateTransform(canvas.width, canvas.height);
        this.fpsCounter = new FPSCounter();
        
        // Rendering options
        this.options = {
            showRoad: true,
            showLanes: true,
            showEgo: true,
            showOthers: true,
            showTrajectories: true
        };
        
        // Line styles
        this.lineWidths = {
            roadBoundary: 2,
            laneLine: 1.5,
            vehicleBorder: 1
        };
        
        this.setupCanvas();
    }
    
    setupCanvas() {
        // Set up high DPI canvas rendering
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;
        
        this.ctx.scale(dpr, dpr);
        this.transform = new CoordinateTransform(rect.width, rect.height);
        
        // Set default drawing styles
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }
    
    resize() {
        this.setupCanvas();
    }
    
    setRenderOptions(options) {
        Object.assign(this.options, options);
    }
    
    clear() {
        this.ctx.fillStyle = Colors.BACKGROUND;
        // Use logical dimensions from transform, not physical canvas dimensions
        // because the context is scaled by DPI
        this.ctx.fillRect(0, 0, this.transform.canvasWidth, this.transform.canvasHeight);
    }
    
    renderFrame(frameData, currentTime = null, trajectoryRenderer = null) {
        this.clear();
        
        // Center view on ego vehicle if available, with offset to show more ahead
        if (frameData.ego_position) {
            // Offset center 10 meters down (in screen space) to show more ahead
            // Since Y-up is forward, we add 10 to centerY to shift view upward
            const forwardOffset = 10; // meters
            this.transform.setCenter(frameData.ego_position.x, frameData.ego_position.y + forwardOffset);
        }
        
        // Draw in order: roads, lanes, trajectories, vehicles
        if (this.options.showRoad) {
            this.drawRoadBoundaries(frameData.road_boundaries);
        }
        
        if (this.options.showLanes) {
            this.drawLaneLines(frameData.lane_lines);
        }
        
        // Draw trajectory predictions if available
        if (this.options.showTrajectories && trajectoryRenderer && currentTime !== null) {
            trajectoryRenderer.renderTrajectories(this.ctx, this.transform, currentTime);
        }
        
        if (this.options.showOthers) {
            this.drawVehicles(frameData.other_vehicles, Colors.OTHER_VEHICLE);
        }
        
        if (this.options.showEgo) {
            this.drawVehicles(frameData.ego_vehicle, Colors.EGO_VEHICLE);
        }
        
        // Update FPS
        this.fpsCounter.update();
    }
    
    drawRoadBoundaries(boundaries) {
        if (!boundaries || boundaries.length === 0) return;
        
        this.ctx.strokeStyle = Colors.ROAD_BOUNDARY;
        this.ctx.lineWidth = this.lineWidths.roadBoundary;
        this.ctx.globalAlpha = Colors.LINE_ALPHA;
        
        this.ctx.beginPath();
        for (const boundary of boundaries) {
            const start = this.transform.worldToScreen(boundary.x0, boundary.y0);
            const end = this.transform.worldToScreen(boundary.x1, boundary.y1);
            
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
        }
        this.ctx.stroke();
        this.ctx.globalAlpha = 1;
    }
    
    drawLaneLines(laneLines) {
        if (!laneLines || laneLines.length === 0) return;
        
        this.ctx.strokeStyle = Colors.LANE_LINE;
        this.ctx.lineWidth = this.lineWidths.laneLine;
        this.ctx.globalAlpha = Colors.LINE_ALPHA;
        
        for (const lane of laneLines) {
            this.ctx.beginPath();
            
            // Draw main line
            const start = this.transform.worldToScreen(lane.x0, lane.y0);
            const end = this.transform.worldToScreen(lane.x1, lane.y1);
            
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
            this.ctx.stroke();
            
            // Draw direction indicator (small perpendicular lines)
            this.drawLaneDirection(lane);
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    drawLaneDirection(lane) {
        // Calculate midpoint and direction
        const midX = (lane.x0 + lane.x1) / 2;
        const midY = (lane.y0 + lane.y1) / 2;
        const heading = lane.heading;
        
        // Draw small perpendicular lines to show direction (flipped and 5x smaller)
        const arrowSize = 0.6; // meters (was 3, now 5x smaller)
        // Flip direction by adding π to the heading
        const flippedHeading = heading + Math.PI;
        const perpAngle1 = flippedHeading + Math.PI / 4;
        const perpAngle2 = flippedHeading - Math.PI / 4;
        
        const arrow1X = midX + arrowSize * Math.cos(perpAngle1);
        const arrow1Y = midY + arrowSize * Math.sin(perpAngle1);
        const arrow2X = midX + arrowSize * Math.cos(perpAngle2);
        const arrow2Y = midY + arrowSize * Math.sin(perpAngle2);
        
        const midScreen = this.transform.worldToScreen(midX, midY);
        const arrow1Screen = this.transform.worldToScreen(arrow1X, arrow1Y);
        const arrow2Screen = this.transform.worldToScreen(arrow2X, arrow2Y);
        
        this.ctx.beginPath();
        this.ctx.moveTo(midScreen.x, midScreen.y);
        this.ctx.lineTo(arrow1Screen.x, arrow1Screen.y);
        this.ctx.moveTo(midScreen.x, midScreen.y);
        this.ctx.lineTo(arrow2Screen.x, arrow2Screen.y);
        this.ctx.stroke();
    }
    
    drawVehicles(vehicles, color) {
        if (!vehicles || vehicles.length === 0) return;
        
        this.ctx.fillStyle = color;
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = this.lineWidths.vehicleBorder;
        this.ctx.globalAlpha = Colors.VEHICLE_ALPHA;
        
        for (const vehicle of vehicles) {
            if (!vehicle.corners || vehicle.corners.length < 4) continue;
            
            this.ctx.beginPath();
            
            // Convert corners to screen coordinates
            const screenCorners = vehicle.corners.map(corner => 
                this.transform.worldToScreen(corner[0], corner[1])
            );
            
            // Draw the polygon
            this.ctx.moveTo(screenCorners[0].x, screenCorners[0].y);
            for (let i = 1; i < screenCorners.length; i++) {
                this.ctx.lineTo(screenCorners[i].x, screenCorners[i].y);
            }
            this.ctx.closePath();
            
            // Fill and stroke
            this.ctx.fill();
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    // Coordinate system methods
    setCenter(worldX, worldY) {
        this.transform.setCenter(worldX, worldY);
    }
    
    setWindowSize(size) {
        this.transform.setWindowSize(size);
    }
    
    setZoom(zoom) {
        this.transform.setZoom(zoom);
    }
    
    pan(deltaScreenX, deltaScreenY) {
        this.transform.pan(deltaScreenX, deltaScreenY);
    }
    
    screenToWorld(screenX, screenY) {
        return this.transform.screenToWorld(screenX, screenY);
    }
    
    worldToScreen(worldX, worldY) {
        return this.transform.worldToScreen(worldX, worldY);
    }
    
    // Debug and utility methods
    drawGrid(gridSize = 10) {
        this.ctx.strokeStyle = '#e0e0e0';
        this.ctx.lineWidth = 0.5;
        this.ctx.globalAlpha = 0.5;
        
        const center = this.transform.worldToScreen(this.transform.centerX, this.transform.centerY);
        const windowSize = this.transform.windowSize;
        
        // Vertical lines
        for (let x = -windowSize; x <= windowSize; x += gridSize) {
            const worldX = this.transform.centerX + x;
            const screenPos = this.transform.worldToScreen(worldX, this.transform.centerY);
            
            this.ctx.beginPath();
            this.ctx.moveTo(screenPos.x, 0);
            this.ctx.lineTo(screenPos.x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = -windowSize; y <= windowSize; y += gridSize) {
            const worldY = this.transform.centerY + y;
            const screenPos = this.transform.worldToScreen(this.transform.centerX, worldY);
            
            this.ctx.beginPath();
            this.ctx.moveTo(0, screenPos.y);
            this.ctx.lineTo(this.canvas.width, screenPos.y);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    drawCrosshair() {
        const center = this.transform.worldToScreen(this.transform.centerX, this.transform.centerY);
        
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.7;
        
        this.ctx.beginPath();
        this.ctx.moveTo(center.x - 10, center.y);
        this.ctx.lineTo(center.x + 10, center.y);
        this.ctx.moveTo(center.x, center.y - 10);
        this.ctx.lineTo(center.x, center.y + 10);
        this.ctx.stroke();
        
        this.ctx.globalAlpha = 1;
    }
    
    getFPS() {
        return this.fpsCounter.fps;
    }
    
    getTransformInfo() {
        return {
            centerX: this.transform.centerX,
            centerY: this.transform.centerY,
            windowSize: this.transform.windowSize,
            zoom: this.transform.zoom,
            scale: this.transform.scale
        };
    }
}

// Export for use in other modules
window.SceneRenderer = SceneRenderer; 