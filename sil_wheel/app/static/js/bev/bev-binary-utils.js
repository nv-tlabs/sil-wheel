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
 * Binary data utilities for handling MessagePack format from the server.
 * 
 * This module provides functions to:
 * 1. Fetch and decode binary data from the server
 * 2. Convert optimized flat arrays back to usable objects
 * 3. Handle both full clips and frame ranges
 * 
 * Expected Binary Clip Data Format:
 * ===================================
 * The server sends MessagePack-encoded data with the following structure:
 * 
 * {
 *   clips: [
 *     {
 *       clip_id: string,              // Unique identifier for the clip
 *       num_frames: integer,          // Total number of frames in the clip
 *       duration: float32,            // Duration in seconds (quantized)
 *       base_timestamp: float32,      // Reference timestamp in seconds (quantized)
 *       t_origin: integer,            // Absolute origin time in microseconds for trajectory alignment
 *       timestamps: float32[],        // Array of relative timestamps (one per frame, quantized)
 *       
 *       // Road boundaries per frame (flat format for efficiency)
 *       road_boundaries: [             // Array with one entry per frame
 *         {
 *           coords: float32[],         // Flat array: [x0, y0, x1, y1, x0, y0, x1, y1, ...]
 *           count: integer             // Number of segments (coords.length / 4)
 *         },
 *         ...
 *       ],
 *       
 *       // Lane lines per frame (same format as road_boundaries)
 *       lane_lines: [                  // Array with one entry per frame
 *         {
 *           coords: float32[],         // Flat array: [x0, y0, x1, y1, x0, y0, x1, y1, ...]
 *           count: integer             // Number of segments (coords.length / 4)
 *         },
 *         ...
 *       ],
 *       
 *       // Ego vehicle per frame (flat format for efficiency)
 *       ego_vehicles: [                // Array with one entry per frame
 *         {
 *           corners: float32[],        // Flat array: [x0, y0, x1, y1, x2, y2, x3, y3]
 *           count: integer             // Number of vehicles (typically 1, corners.length / 8)
 *         },
 *         ...
 *       ],
 *       
 *       // Other vehicles per frame (same format as ego_vehicles)
 *       other_vehicles: [              // Array with one entry per frame
 *         {
 *           corners: float32[],        // Flat array: [x0, y0, x1, y1, x2, y2, x3, y3, ...]
 *           count: integer             // Number of vehicles (corners.length / 8)
 *         },
 *         ...
 *       ],
 *       
 *       // Ego positions for all frames (flat format)
 *       ego_positions: float32[]       // Flat array: [x0, y0, x1, y1, ...] for all frames
 *     }
 *   ],
 *   batch_size: integer                // Number of clips in the batch
 * }
 * 
 * The flat array formats significantly reduce MessagePack payload size by avoiding
 * nested object structures. This class handles converting these flat formats back
 * to the nested object structure expected by the renderer.
 */

// Import msgpack library (you'll need to include this in your HTML)
// <script src="https://cdn.jsdelivr.net/npm/@msgpack/msgpack@2.8.0/dist.umd/index.min.js"></script>

class BinaryDataHandler {
    constructor() {
        this.cache = new Map();
    }

    /**
     * Fetch binary data from the server
     */
    async fetchBinaryData(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const arrayBuffer = await response.arrayBuffer();
            const uint8Array = new Uint8Array(arrayBuffer);
            
            // Decode MessagePack data
            const data = MessagePack.decode(uint8Array);
            return data;
        } catch (error) {
            console.error('Error fetching binary data:', error);
            throw error;
        }
    }

    /**
     * Fetch full clip data in binary format
     */
    async fetchClipData(clipId = null) {
        const url = clipId 
            ? `/api/clip?clip_id=${clipId}&format=binary`
            : '/api/clip?format=binary';
        
        return await this.fetchBinaryData(url);
    }

    /**
     * Fetch frame range in binary format
     */
    async fetchFrameRange(startFrame, endFrame, clipId = null) {
        const url = clipId
            ? `/api/clip/frames/${startFrame}/${endFrame}/binary?clip_id=${clipId}`
            : `/api/clip/frames/${startFrame}/${endFrame}/binary`;
        
        return await this.fetchBinaryData(url);
    }

    /**
     * Convert optimized segment data back to usable format
     */
    convertSegmentsFromFlat(segmentData) {
        const segments = [];
        const coords = segmentData.coords;
        const count = segmentData.count;

        for (let i = 0; i < count; i++) {
            const baseIdx = i * 4;
            segments.push({
                x0: coords[baseIdx],
                y0: coords[baseIdx + 1],
                x1: coords[baseIdx + 2],
                y1: coords[baseIdx + 3]
            });
        }

        return segments;
    }

    /**
     * Convert optimized vehicle data back to usable format
     */
    convertVehiclesFromFlat(vehicleData) {
        const vehicles = [];
        const corners = vehicleData.corners;
        const count = vehicleData.count;

        for (let i = 0; i < count; i++) {
            const baseIdx = i * 8; // 4 corners * 2 coordinates
            const vehicleCorners = [];
            
            for (let j = 0; j < 4; j++) {
                const cornerIdx = baseIdx + j * 2;
                vehicleCorners.push([
                    corners[cornerIdx],
                    corners[cornerIdx + 1]
                ]);
            }
            
            vehicles.push({
                corners: vehicleCorners
            });
        }

        return vehicles;
    }

    /**
     * Convert ego positions from flat array
     */
    convertEgoPositions(egoPositions, frameIndex) {
        const baseIdx = frameIndex * 2;
        return {
            x: egoPositions[baseIdx],
            y: egoPositions[baseIdx + 1]
        };
    }

    /**
     * Convert binary clip data to the format expected by the renderer
     */
    convertBinaryClipToRenderFormat(binaryClipData) {
        const clip = binaryClipData.clips[0]; // Assuming single clip
        
        const frames = [];
        for (let i = 0; i < clip.num_frames; i++) {
            const frame = {
                timestamp: clip.base_timestamp + clip.timestamps[i],
                time_index: i,
                
                // Convert road boundaries
                road_boundaries: this.convertSegmentsFromFlat(clip.road_boundaries[i]),
                
                // Convert lane lines
                lane_lines: this.convertSegmentsFromFlat(clip.lane_lines[i]),
                
                // Convert ego vehicle
                ego_vehicle: this.convertVehiclesFromFlat(clip.ego_vehicles[i]),
                
                // Convert other vehicles
                other_vehicles: this.convertVehiclesFromFlat(clip.other_vehicles[i]),
                
                // Convert ego position
                ego_position: this.convertEgoPositions(clip.ego_positions, i)
            };
            
            frames.push(frame);
        }

        return {
            clips: [{
                clip_id: clip.clip_id,
                num_frames: clip.num_frames,
                duration: clip.duration,
                video_offset: clip.video_offset || 0,  // Offset between video start and clip data start
                t_origin: clip.t_origin,  // Absolute origin time in microseconds
                base_timestamp: clip.base_timestamp,  // Base timestamp in seconds
                frames: frames
            }],
            batch_size: binaryClipData.batch_size
        };
    }

    /**
     * Get compression statistics
     */
    async getCompressionStats(clipId = null) {
        const url = clipId 
            ? `/api/clip/compression-stats?clip_id=${clipId}`
            : '/api/clip/compression-stats';
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Progressive loading: load frames in chunks
     */
    async loadFramesProgressively(totalFrames, chunkSize = 50, clipId = null, onChunkLoaded = null) {
        const chunks = [];
        
        for (let start = 0; start < totalFrames; start += chunkSize) {
            const end = Math.min(start + chunkSize - 1, totalFrames - 1);
            
            try {
                const chunkData = await this.fetchFrameRange(start, end, clipId);
                const convertedChunk = this.convertBinaryClipToRenderFormat({
                    clips: [chunkData],
                    batch_size: 1
                });
                
                chunks.push(convertedChunk);
                
                if (onChunkLoaded) {
                    onChunkLoaded(convertedChunk, start, end);
                }
                
            } catch (error) {
                console.error(`Error loading chunk ${start}-${end}:`, error);
                throw error;
            }
        }
        
        return chunks;
    }
}

// Export for use in other modules
window.BinaryDataHandler = BinaryDataHandler; 