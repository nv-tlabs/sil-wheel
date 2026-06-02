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
 * Utility functions for coordinate transformations and common operations.
 */

class CoordinateTransform {
    constructor(canvasWidth, canvasHeight) {
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.centerX = 0;  // World center X
        this.centerY = 0;  // World center Y
        this.scale = 1;    // Pixels per meter
        this.windowSize = 50; // Viewing window size in meters
        this.zoom = 1;     // Zoom factor
        
        this.updateScale();
    }
    
    updateScale() {
        // Calculate scale to fit the window in the canvas
        this.scale = Math.min(this.canvasWidth, this.canvasHeight) / (2 * this.windowSize) * this.zoom;
    }
    
    setCenter(worldX, worldY) {
        this.centerX = worldX;
        this.centerY = worldY;
    }
    
    setWindowSize(size) {
        this.windowSize = size;
        this.updateScale();
    }
    
    setZoom(zoom) {
        this.zoom = zoom;
        this.updateScale();
    }
    
    worldToScreen(worldX, worldY) {
        const screenX = (worldX - this.centerX) * this.scale + this.canvasWidth / 2;
        const screenY = -(worldY - this.centerY) * this.scale + this.canvasHeight / 2; // Negative Y for screen coordinates
        return { x: screenX, y: screenY };
    }
    
    screenToWorld(screenX, screenY) {
        const worldX = (screenX - this.canvasWidth / 2) / this.scale + this.centerX;
        const worldY = -((screenY - this.canvasHeight / 2) / this.scale) + this.centerY; // Negative Y for world coordinates
        return { x: worldX, y: worldY };
    }
    
    pan(deltaScreenX, deltaScreenY) {
        const deltaWorldX = deltaScreenX / this.scale;
        const deltaWorldY = -deltaScreenY / this.scale; // Negative Y for world coordinates
        this.centerX -= deltaWorldX;
        this.centerY -= deltaWorldY;
    }
}

class FPSCounter {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
        this.updateInterval = 500; // Update every 500ms
    }
    
    update() {
        this.frameCount++;
        const now = performance.now();
        const elapsed = now - this.lastTime;
        
        if (elapsed >= this.updateInterval) {
            this.fps = Math.round((this.frameCount * 1000) / elapsed);
            this.frameCount = 0;
            this.lastTime = now;
        }
        
        return this.fps;
    }
}

class APIClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.binaryHandler = null; // Initialize later when BinaryDataHandler is available
        this.preferBinary = true;
    }
    
    async waitForMessagePack() {
        // Wait for MessagePack to be available
        if (typeof MessagePack !== 'undefined') {
            return true;
        }
        
        return new Promise((resolve) => {
            const checkInterval = setInterval(() => {
                if (window.messagePack_ready || typeof MessagePack !== 'undefined') {
                    clearInterval(checkInterval);
                    resolve(true);
                }
            }, 50);
            
            // Fallback to JSON after 3 seconds
            setTimeout(() => {
                clearInterval(checkInterval);
                console.warn('MessagePack not available, falling back to JSON format');
                this.preferBinary = false;
                resolve(false);
            }, 3000);
        });
    }
    
    async get(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Check if response is binary (MessagePack)
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/octet-stream')) {
                // Ensure MessagePack is available
                const hasMessagePack = await this.waitForMessagePack();
                if (hasMessagePack) {
                    const arrayBuffer = await response.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);
                    return MessagePack.decode(uint8Array);
                } else {
                    throw new Error('Binary response received but MessagePack not available');
                }
            } else {
                // Handle JSON response
                return await response.json();
            }
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }
    
    async getBinary(endpoint) {
        try {
            // Ensure MessagePack is available
            const hasMessagePack = await this.waitForMessagePack();
            if (!hasMessagePack) {
                throw new Error('MessagePack library not available for binary format');
            }
            
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const arrayBuffer = await response.arrayBuffer();
            const uint8Array = new Uint8Array(arrayBuffer);
            return MessagePack.decode(uint8Array);
        } catch (error) {
            console.error(`Binary API request failed for ${endpoint}:`, error);
            throw error;
        }
    }
    
    async getWithFallback(endpoint) {
        try {
            // Try binary format first if MessagePack is available
            const hasMessagePack = await this.waitForMessagePack();
            if (hasMessagePack && this.preferBinary) {
                const binaryEndpoint = endpoint.includes('?') 
                    ? `${endpoint}&format=binary` 
                    : `${endpoint}?format=binary`;
                return await this.getBinary(binaryEndpoint);
            } else {
                // Fallback to JSON format
                const jsonEndpoint = endpoint.includes('?') 
                    ? `${endpoint}&format=json` 
                    : `${endpoint}?format=json`;
                return await this.get(jsonEndpoint);
            }
        } catch (error) {
            console.error(`Request with fallback failed for ${endpoint}:`, error);
            throw error;
        }
    }
    
    async post(endpoint, data = {}) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }
}

// Utility functions
function formatTime(seconds) {
    return `${seconds.toFixed(1)}s`;
}

function formatPosition(x, y) {
    return `(${x.toFixed(1)}, ${y.toFixed(1)})`;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Color utilities for consistent visualization
const Colors = {
    ROAD_BOUNDARY: '#000000',     // Black
    LANE_LINE: '#ff8c00',         // Dark orange
    EGO_VEHICLE: '#32cd32',       // Lime green
    OTHER_VEHICLE: '#1e90ff',     // Dodger blue
    BACKGROUND: '#fafafa',        // Light gray
    
    // Transparencies
    VEHICLE_ALPHA: 0.8,
    LINE_ALPHA: 1.0
};

// Export for use in other modules
window.CoordinateTransform = CoordinateTransform;
window.FPSCounter = FPSCounter;
window.APIClient = APIClient;
window.Colors = Colors;
window.formatTime = formatTime;
window.formatPosition = formatPosition;
window.clamp = clamp;
window.lerp = lerp;
window.debounce = debounce; 