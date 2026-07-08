let session;
let isRecording = false;
let recordedData = [];
let startTime = 0;
let animationId;
let videoElement = document.getElementById('webcam');
let canvasElement = document.getElementById('overlay');
let ctx = canvasElement.getContext('2d');
let statusText = document.getElementById('statusText');
let fpsCounter = document.getElementById('fpsCounter');
let loadingOverlay = document.getElementById('loading');
let startBtn = document.getElementById('startBtn');
let endBtn = document.getElementById('endBtn');

let lastFrameTime = 0;
let frameCount = 0;

// Model input size for RTMPose is usually 192x256
const inputWidth = 192;
const inputHeight = 256;

// Create an offscreen canvas for preprocessing
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = inputWidth;
offscreenCanvas.height = inputHeight;
const offCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

async function init() {
    try {
        // 1. Setup Camera
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
        });
        videoElement.srcObject = stream;
        
        await new Promise(resolve => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                resolve();
            };
        });

        // 2. Load ONNX Model
        statusText.innerText = "Loading model...";
        // Set wasm threads
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        ort.env.wasm.simd = true;
        session = await ort.InferenceSession.create('models/rtmpose.onnx', { executionProviders: ['webgpu', 'wasm'] });
        console.log("Model loaded", session.inputNames, session.outputNames);
        
        loadingOverlay.classList.add('hidden');
        statusText.innerText = "Ready";
        startBtn.disabled = false;
        
    } catch (e) {
        console.error(e);
        statusText.innerText = "Error: " + e.message;
        document.getElementById('loadingText').innerText = "Error loading model or camera. Check console.";
    }
}

function preprocess(video) {
    // Draw video frame to offscreen canvas
    offCtx.drawImage(video, 0, 0, inputWidth, inputHeight);
    const imageData = offCtx.getImageData(0, 0, inputWidth, inputHeight).data;
    
    // Convert to Float32Array and normalize
    const float32Data = new Float32Array(3 * inputHeight * inputWidth);
    
    // Normalization parameters (ImageNet)
    const mean = [123.675, 116.28, 103.53];
    const std = [58.395, 57.12, 57.375];

    for (let i = 0; i < inputHeight * inputWidth; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];

        // Format is NCHW (1, 3, H, W)
        float32Data[i] = (r - mean[0]) / std[0]; // R
        float32Data[inputHeight * inputWidth + i] = (g - mean[1]) / std[1]; // G
        float32Data[2 * inputHeight * inputWidth + i] = (b - mean[2]) / std[2]; // B
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, inputHeight, inputWidth]);
}

async function processFrame() {
    if (!isRecording) return;
    
    const startTs = performance.now();
    
    try {
        const t0 = performance.now();
        const inputTensor = preprocess(videoElement);
        const t1 = performance.now();
        
        // Run inference
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;
        const results = await session.run(feeds);
        const t2 = performance.now();
        
        // Parse outputs for SimCC format
        let keypoints = [];
        const simccX = results['simcc_x'].data; // [1, 17, 384]
        const simccY = results['simcc_y'].data; // [1, 17, 512]
        
        for (let i = 0; i < 17; i++) {
            // Find argmax for X
            let maxX = -Infinity;
            let maxIdxX = 0;
            for (let j = 0; j < 384; j++) {
                const val = simccX[i * 384 + j];
                if (val > maxX) {
                    maxX = val;
                    maxIdxX = j;
                }
            }
            
            // Find argmax for Y
            let maxY = -Infinity;
            let maxIdxY = 0;
            for (let j = 0; j < 512; j++) {
                const val = simccY[i * 512 + j];
                if (val > maxY) {
                    maxY = val;
                    maxIdxY = j;
                }
            }
            
            // SimCC scales input dimensions by 2 (192*2=384, 256*2=512)
            const px = ((maxIdxX / 2.0) / inputWidth) * canvasElement.width;
            const py = ((maxIdxY / 2.0) / inputHeight) * canvasElement.height;
            keypoints.push([px, py]);
        }
        
        // Save to array
        recordedData.push({
            timestamp: Date.now() - startTime,
            keypoints: keypoints
        });
        
        // Draw Skeleton
        const skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
            [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], 
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], 
            [3, 5], [4, 6]
        ];
        
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Draw lines
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        for (let edge of skeleton) {
            const p1 = keypoints[edge[0]];
            const p2 = keypoints[edge[1]];
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1[0], p1[1]);
                ctx.lineTo(p2[0], p2[1]);
                ctx.stroke();
            }
        }
        
        // Draw points
        ctx.fillStyle = '#10b981';
        for (let pt of keypoints) {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 6, 0, 2 * Math.PI);
            ctx.fill();
        }
        const t3 = performance.now();
        
        // Calculate FPS and Performance
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            fpsCounter.innerText = `FPS: ${frameCount} | Prep: ${(t1-t0).toFixed(1)}ms | Inf: ${(t2-t1).toFixed(1)}ms | Draw: ${(t3-t2).toFixed(1)}ms`;
            frameCount = 0;
            lastFrameTime = now;
        }
        
    } catch(e) {
        console.error("Inference error:", e);
    }
    
    animationId = requestAnimationFrame(processFrame);
}

startBtn.addEventListener('click', () => {
    isRecording = true;
    recordedData = [];
    startTime = Date.now();
    startBtn.disabled = true;
    endBtn.disabled = false;
    statusText.innerText = "Recording...";
    processFrame();
});

endBtn.addEventListener('click', () => {
    isRecording = false;
    cancelAnimationFrame(animationId);
    startBtn.disabled = false;
    endBtn.disabled = true;
    statusText.innerText = "Ready";
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Save JSON
    const dataStr = JSON.stringify(recordedData, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `pose_data_${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

init();
