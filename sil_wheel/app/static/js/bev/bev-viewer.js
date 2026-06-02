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
 * BEV Tile Manager - Manages BEV canvas instances for multiple clip tiles
 * 
 * This module handles:
 * - Creating and managing BEV canvas instances per clip
 * - Loading BEV data asynchronously from the backend
 * - Synchronizing BEV visualization with video playback
 * - Handling video_offset for correct time alignment
 */

class BEVInstance {
    constructor(clipId, container, videoElement) {
        this.clipId = clipId;
        this.container = container;
        this.videoElement = videoElement;
        this.canvas = null;
        this.renderer = null;
        this.clipData = null;
        this.isLoading = false;
        this.hasError = false;
        this.videoOffset = 0;
        this.lastSyncTime = 0;
        
        // Mouse interaction state
        this.isDragging = false;
        this.lastMousePos = { x: 0, y: 0 };
        this.currentZoom = 1.0;
        this.isEgoCentered = true;
        
        // Store bound event handlers for cleanup
        this.mouseDownHandler = (e) => this.onMouseDown(e);
        this.mouseMoveHandler = (e) => this.onMouseMove(e);
        this.mouseUpHandler = (e) => this.onMouseUp(e);
        this.wheelHandler = (e) => this.onWheel(e);
        this.dblClickHandler = (e) => this.onDoubleClick(e);
        
        this.initializeUI();
        this.setupMouseInteractions();
    }
    
    initializeUI() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'bev-canvas';
        this.canvas.width = 800;
        this.canvas.height = 600;
        this.container.appendChild(this.canvas);
        
        // Create loading indicator
        this.loadingEl = document.createElement('div');
        this.loadingEl.className = 'bev-loading';
        this.loadingEl.textContent = 'Loading BEV data...';
        this.loadingEl.style.display = 'none';
        this.container.appendChild(this.loadingEl);
        
        // Create message element
        this.messageEl = document.createElement('div');
        this.messageEl.className = 'bev-message';
        this.messageEl.style.display = 'none';
        this.container.appendChild(this.messageEl);
        
        // Initialize renderer
        this.renderer = new SceneRenderer(this.canvas);
    }
    
    async loadBEVData() {
        if (this.isLoading || this.clipData) {
            return;
        }
        
        this.isLoading = true;
        this.showLoading();
        
        try {
            const binaryHandler = new BinaryDataHandler();
            const binaryData = await binaryHandler.fetchBinaryData(`/api/bev/${this.clipId}`);
            
            // Convert binary data to render format
            this.clipData = binaryHandler.convertBinaryClipToRenderFormat(binaryData);
            
            if (this.clipData && this.clipData.clips && this.clipData.clips.length > 0) {
                const clip = this.clipData.clips[0];
                this.videoOffset = clip.video_offset || 0;
                console.log(`BEV loaded for ${this.clipId}: ${clip.num_frames} frames, video_offset=${this.videoOffset}s`);
                
                this.hideLoading();
                this.hideMessage();
                
                // Initial sync
                this.syncWithVideo();
            } else {
                throw new Error('Invalid BEV data format');
            }
        } catch (error) {
            console.error(`Error loading BEV data for ${this.clipId}:`, error);
            this.hasError = true;
            this.hideLoading();
            this.showMessage('No BEV data available');
        } finally {
            this.isLoading = false;
        }
    }
    
    syncWithVideo() {
        if (!this.clipData || !this.videoElement) {
            return;
        }
        
        const videoTime = this.videoElement.currentTime;
        
        // Convert video time to BEV data time
        // bevDataTime = videoTime + videoOffset
        const bevDataTime = videoTime + this.videoOffset;
        
        const clip = this.clipData.clips[0];
        
        // Check if we're outside the BEV data range
        if (bevDataTime < 0 || bevDataTime > clip.duration) {
            // Show empty canvas with message
            this.showMessage('No BEV data at this timestamp');
            this.renderer.clear();
            return;
        }
        
        this.hideMessage();
        
        // Find the appropriate frame for this BEV data time
        let targetFrameIndex = 0;
        for (let i = 0; i < clip.frames.length; i++) {
            const frameTime = clip.frames[i].timestamp - clip.base_timestamp;
            if (frameTime <= bevDataTime) {
                targetFrameIndex = i;
            } else {
                break;
            }
        }
        
        // Render the frame
        const frame = clip.frames[targetFrameIndex];
        
        // If not ego-centered, preserve current view transform
        if (!this.isEgoCentered) {
            // Store current transform before rendering
            const currentTransform = {
                centerX: this.renderer.transform.centerX,
                centerY: this.renderer.transform.centerY,
                zoom: this.renderer.transform.zoom
            };
            
            this.renderer.renderFrame(frame, frame.timestamp, null);
            
            // Restore transform (renderFrame may have reset it)
            this.renderer.transform.centerX = currentTransform.centerX;
            this.renderer.transform.centerY = currentTransform.centerY;
            this.renderer.setZoom(currentTransform.zoom);
            
            // Re-render with preserved transform
            this.renderer.renderFrame(frame, frame.timestamp, null);
        } else {
            this.renderer.renderFrame(frame, frame.timestamp, null);
        }
    }
    
    setupMouseInteractions() {
        if (!this.canvas) return;
        
        // Set cursor style
        this.canvas.style.cursor = 'grab';
        
        // Mouse event listeners
        this.canvas.addEventListener('mousedown', this.mouseDownHandler);
        this.canvas.addEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.addEventListener('mouseup', this.mouseUpHandler);
        this.canvas.addEventListener('mouseleave', this.mouseUpHandler);
        this.canvas.addEventListener('wheel', this.wheelHandler, { passive: false });
        this.canvas.addEventListener('dblclick', this.dblClickHandler);
    }
    
    onMouseDown(e) {
        this.isDragging = true;
        this.lastMousePos = this.getMousePos(e);
        this.canvas.style.cursor = 'grabbing';
    }
    
    onMouseMove(e) {
        const mousePos = this.getMousePos(e);
        
        if (this.isDragging) {
            const deltaX = mousePos.x - this.lastMousePos.x;
            const deltaY = mousePos.y - this.lastMousePos.y;
            
            this.renderer.pan(deltaX, deltaY);
            this.syncWithVideo(); // Re-render with new pan
            this.isEgoCentered = false;
            
            this.lastMousePos = mousePos;
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = 'grab';
    }
    
    onWheel(e) {
        e.preventDefault();
        
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        this.currentZoom = clamp(this.currentZoom * zoomFactor, 0.1, 3.0);
        
        this.renderer.setZoom(this.currentZoom);
        this.syncWithVideo(); // Re-render with new zoom
        this.isEgoCentered = false;
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    onDoubleClick(e) {
        // Reset view to default: centered on ego with default zoom
        this.currentZoom = 1.0;
        this.isEgoCentered = true;
        this.renderer.setZoom(this.currentZoom);
        this.syncWithVideo();
    }
    
    showLoading() {
        if (this.loadingEl) {
            this.loadingEl.style.display = 'block';
        }
    }
    
    hideLoading() {
        if (this.loadingEl) {
            this.loadingEl.style.display = 'none';
        }
    }
    
    showMessage(text) {
        if (this.messageEl) {
            this.messageEl.textContent = text;
            this.messageEl.style.display = 'block';
        }
    }
    
    hideMessage() {
        if (this.messageEl) {
            this.messageEl.style.display = 'none';
        }
    }
    
    destroy() {
        // Remove mouse event listeners
        if (this.canvas) {
            this.canvas.removeEventListener('mousedown', this.mouseDownHandler);
            this.canvas.removeEventListener('mousemove', this.mouseMoveHandler);
            this.canvas.removeEventListener('mouseup', this.mouseUpHandler);
            this.canvas.removeEventListener('mouseleave', this.mouseUpHandler);
            this.canvas.removeEventListener('wheel', this.wheelHandler);
            this.canvas.removeEventListener('dblclick', this.dblClickHandler);
        }
        
        // Clean up resources
        if (this.container) {
            this.container.innerHTML = '';
        }
        this.clipData = null;
        this.renderer = null;
        this.canvas = null;
    }
}

class BEVTileManager {
    constructor() {
        this.bevInstances = new Map(); // clipId -> BEVInstance
    }
    
    /**
     * Initialize BEV for a clip tile
     */
    async initializeBEV(clipId, container, videoElement) {
        // Check if already initialized
        if (this.bevInstances.has(clipId)) {
            return this.bevInstances.get(clipId);
        }
        
        // Create new instance
        const instance = new BEVInstance(clipId, container, videoElement);
        this.bevInstances.set(clipId, instance);
        
        // Load data asynchronously
        await instance.loadBEVData();
        
        // Set up video time update listener
        if (videoElement) {
            // Use a throttled sync function to avoid excessive updates
            const throttledSync = () => {
                const now = Date.now();
                if (now - instance.lastSyncTime > 100) { // Throttle to ~10 fps
                    instance.syncWithVideo();
                    instance.lastSyncTime = now;
                }
            };
            
            videoElement.addEventListener('timeupdate', throttledSync);
            videoElement.addEventListener('seeked', () => instance.syncWithVideo());
            
            // Store listeners for cleanup
            instance.syncListener = throttledSync;
            instance.seekListener = () => instance.syncWithVideo();
        }
        
        return instance;
    }
    
    /**
     * Sync BEV with video time (called from video timeupdate events)
     */
    syncBEVWithVideo(clipId) {
        const instance = this.bevInstances.get(clipId);
        if (instance) {
            instance.syncWithVideo();
        }
    }
    
    /**
     * Destroy BEV instance for a clip
     */
    destroyBEV(clipId) {
        const instance = this.bevInstances.get(clipId);
        if (instance) {
            // Remove event listeners
            if (instance.videoElement && instance.syncListener) {
                instance.videoElement.removeEventListener('timeupdate', instance.syncListener);
                instance.videoElement.removeEventListener('seeked', instance.seekListener);
            }
            
            instance.destroy();
            this.bevInstances.delete(clipId);
        }
    }
    
    /**
     * Destroy all BEV instances
     */
    destroyAll() {
        for (const clipId of this.bevInstances.keys()) {
            this.destroyBEV(clipId);
        }
    }
}

// Create global instance
window.bevTileManager = new BEVTileManager();

// Export for use in other modules
window.BEVTileManager = BEVTileManager;
window.BEVInstance = BEVInstance;

