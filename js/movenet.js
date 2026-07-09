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

// --- FSM & Smoothing Variables ---
const STATES = {
    CALIBRATING: 0,
    WAITING_TO_STAND: 1,
    WALKING_AWAY: 2,
    WALKING_BACK: 3,
    SITTING_DOWN: 4,
    DONE: 5
};
const STATE_NAMES = ["КАЛІБРУВАННЯ", "ОЧІКУВАННЯ РУХУ", "ВІДДАЛЕННЯ", "ПОВЕРНЕННЯ", "СІДАННЯ", "ЗАВЕРШЕНО"];

let currentState = STATES.CALIBRATING;
let calibrationData = { baseHipsY: 0, baseTorsoLength: 0, startX: 0, samples: 0 };
let stateTimer = 0;
let maxDistance = 0;

let smoothedHipsX = 0;
let smoothedHipsY = 0;
let smoothedTorso = 0;
const EMA_ALPHA = 0.2; // Smoothing factor

// --- Voice & Audio Helpers ---
function speak(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'uk-UA';
    speechSynthesis.speak(utterance);
}

function playBeep(frequency = 440, duration = 200) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;
    
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    oscillator.start();
    gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + duration / 1000);
    
    setTimeout(() => {
        oscillator.stop();
        audioCtx.close();
    }, duration);
}

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
    
    // Global 30-second timeout check
    if (Date.now() - startTime > 30000 && currentState !== STATES.DONE) {
        playBeep(300, 500);
        speak("Таймаут. Запис скасовано.");
        console.warn("Global timeout reached");
        endBtn.click();
        return;
    }

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
            
            // --- FSM Logic ---
            const leftHip = keypoints[11];
            const rightHip = keypoints[12];
            const leftShoulder = keypoints[5];
            const rightShoulder = keypoints[6];

            if (leftHip.score > 0.3 && rightHip.score > 0.3 && leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
                const currentHipsY = (leftHip.y + rightHip.y) / 2;
                const currentHipsX = (leftHip.x + rightHip.x) / 2;
                const currentShouldersY = (leftShoulder.y + rightShoulder.y) / 2;
                const currentTorso = currentHipsY - currentShouldersY; // Y goes down

                if (smoothedHipsY === 0) {
                    smoothedHipsY = currentHipsY;
                    smoothedHipsX = currentHipsX;
                    smoothedTorso = currentTorso;
                } else {
                    smoothedHipsY = EMA_ALPHA * currentHipsY + (1 - EMA_ALPHA) * smoothedHipsY;
                    smoothedHipsX = EMA_ALPHA * currentHipsX + (1 - EMA_ALPHA) * smoothedHipsX;
                    smoothedTorso = EMA_ALPHA * currentTorso + (1 - EMA_ALPHA) * smoothedTorso;
                }

                const timeSinceStart = Date.now() - startTime;
                const distFromStart = Math.abs(smoothedHipsX - calibrationData.startX);
                // Assume threshold is 15% of torso length for standing detection
                const standThreshold = calibrationData.baseHipsY - (calibrationData.baseTorsoLength * 0.15); 

                switch (currentState) {
                    case STATES.CALIBRATING:
                        calibrationData.baseHipsY += smoothedHipsY;
                        calibrationData.startX += smoothedHipsX;
                        calibrationData.baseTorsoLength += smoothedTorso;
                        calibrationData.samples++;
                        
                        if (timeSinceStart > 1500) { // 1.5s calibration
                            calibrationData.baseHipsY /= calibrationData.samples;
                            calibrationData.startX /= calibrationData.samples;
                            calibrationData.baseTorsoLength /= calibrationData.samples;
                            
                            currentState = STATES.WAITING_TO_STAND;
                            speak("Встаньте і пройдіть 3 метри");
                        }
                        break;

                    case STATES.WAITING_TO_STAND:
                        // Y coordinate decreases when standing up (moves up on screen)
                        if (smoothedHipsY < standThreshold) {
                            playBeep(600, 300); // Signal movement start
                            currentState = STATES.WALKING_AWAY;
                            maxDistance = 0;
                        }
                        break;

                    case STATES.WALKING_AWAY:
                        if (distFromStart > maxDistance) {
                            maxDistance = distFromStart;
                        }
                        // Turn detection: max distance reached (at least 30px) and now going back
                        if (maxDistance > 30 && distFromStart < maxDistance - 15) {
                            speak("Поверніться і сядьте");
                            currentState = STATES.WALKING_BACK;
                        }
                        break;

                    case STATES.WALKING_BACK:
                        // Returned close to start X, and hips lowered again
                        if (distFromStart < 50 && smoothedHipsY > standThreshold) {
                            currentState = STATES.SITTING_DOWN;
                            stateTimer = Date.now();
                        }
                        break;

                    case STATES.SITTING_DOWN:
                        if (smoothedHipsY < standThreshold) {
                            // False alarm, stood up again
                            currentState = STATES.WALKING_BACK;
                        } else if (Date.now() - stateTimer > 1000) {
                            playBeep(800, 400); // Signal end
                            speak("Тест завершено");
                            currentState = STATES.DONE;
                            endBtn.click(); // Auto-stop
                        }
                        break;
                }
            }

            // Draw State UI overlay
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillRect(10, 10, 250, 40);
            ctx.fillStyle = '#fff';
            ctx.font = '20px Outfit, sans-serif';
            ctx.fillText(`Стан: ${STATE_NAMES[currentState]}`, 20, 38);
            
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
    
    // Reset FSM
    currentState = STATES.CALIBRATING;
    calibrationData = { baseHipsY: 0, baseTorsoLength: 0, startX: 0, samples: 0 };
    smoothedHipsX = 0;
    smoothedHipsY = 0;
    smoothedTorso = 0;
    maxDistance = 0;
    
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
