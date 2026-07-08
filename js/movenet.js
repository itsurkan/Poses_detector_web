let detector;
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

// MoveNet COCO connections (17 keypoints)
const skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], 
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], 
    [3, 5], [4, 6]
];

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

        // 2. Load MoveNet Thunder Model
        statusText.innerText = "Loading MoveNet...";
        await tf.setBackend('webgl');
        await tf.ready();
        
        const detectorConfig = { modelType: poseDetection.movenet.modelType.THUNDER };
        detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
        
        console.log("MoveNet loaded!");
        
        loadingOverlay.classList.add('hidden');
        statusText.innerText = "Ready";
        startBtn.disabled = false;
        
    } catch (e) {
        console.error(e);
        statusText.innerText = "Error: " + e.message;
        document.getElementById('loadingText').innerText = "Error loading model or camera.";
    }
}

async function processFrame() {
    if (!isRecording) return;
    
    const t0 = performance.now();
    
    try {
        // Run inference
        const poses = await detector.estimatePoses(videoElement, {
            maxPoses: 1,
            flipHorizontal: false // We mirror in CSS
        });
        const t1 = performance.now();
        
        let keypointsArray = [];
        
        // Draw
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        if (poses.length > 0) {
            const keypoints = poses[0].keypoints;
            
            // Format for JSON storage
            for (let i = 0; i < 17; i++) {
                keypointsArray.push([keypoints[i].x, keypoints[i].y]);
            }
            
            // Save to array
            recordedData.push({
                timestamp: Date.now() - startTime,
                keypoints: keypointsArray
            });
            
            // Draw lines
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 3;
            for (let edge of skeleton) {
                const p1 = keypoints[edge[0]];
                const p2 = keypoints[edge[1]];
                if (p1.score > 0.3 && p2.score > 0.3) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            }
            
            // Draw points
            ctx.fillStyle = '#10b981';
            for (let pt of keypoints) {
                if (pt.score > 0.3) {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }
        }
        
        const t2 = performance.now();
        
        // Calculate FPS and Performance
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            fpsCounter.innerText = `FPS: ${frameCount} | Inf: ${(t1-t0).toFixed(1)}ms | Draw: ${(t2-t1).toFixed(1)}ms`;
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
    a.download = `pose_data_movenet_${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

init();
