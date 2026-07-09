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
let stateLabel = document.getElementById('stateLabel');

let lastFrameTime = 0;
let frameCount = 0;

// --- FSM & Smoothing Variables ---
const STATES = {
    PRE_CALIBRATION: 0,
    CALIBRATING: 1,
    WAITING_TO_STAND: 2,
    WALKING_AWAY: 3,
    WALKING_BACK: 4,
    SITTING_DOWN: 5,
    DONE: 6
};
const STATE_NAMES = ["ПІДГОТОВКА", "КАЛІБРУВАННЯ", "ОЧІКУВАННЯ РУХУ", "ВІДДАЛЕННЯ", "ПОВЕРНЕННЯ", "СІДАННЯ", "ЗАВЕРШЕНО"];

let currentState = STATES.CALIBRATING;

// The state label lives in the DOM (not the mirrored canvas), so the text
// stays readable while the video/skeleton stay flipped.
function setState(newState) {
    currentState = newState;
    stateLabel.textContent = STATE_NAMES[newState];
}

let calibrationData = { baseHipsY: 0, baseTorsoLength: 0, startX: 0, samples: 0 };
let calibrationStartedAt = 0;
let goSignalAt = 0;       // when the "stand up and walk" instruction was given
let lastSitPromptAt = 0;  // re-prompt throttle while waiting for the person to sit
let lastSitDownCueAt = 0; // cue throttle while the person hovers over the chair
let stateTimer = 0;
let maxDistance = 0;

let smoothedHipsX = 0;
let smoothedHipsY = 0;
let smoothedTorso = 0;
let smoothedKneesY = 0;
let kneesVisible = false;
const EMA_ALPHA = 0.2; // Smoothing factor

// Seated-pose check: sitting puts the knees roughly at hip level, standing puts
// the hips a full thigh-length above the knees. Returns null when the knees are
// not tracked, so callers can degrade to the hips-threshold-only behaviour.
function isSeated() {
    if (!kneesVisible || smoothedTorso <= 0) return null;
    return (smoothedKneesY - smoothedHipsY) < smoothedTorso * 0.35;
}

// --- Voice & Audio Helpers ---
// Web Speech API rarely ships a Ukrainian voice, so instructions are spoken
// in English with an explicitly selected en-US voice.
let audioCtx = null;

function getAudioCtx() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function pickEnglishVoice() {
    const voices = speechSynthesis.getVoices();
    return voices.find(v => v.lang === 'en-US' && /Google/i.test(v.name))
        || voices.find(v => v.lang === 'en-US')
        || voices.find(v => v.lang && v.lang.startsWith('en'))
        || null;
}

function speak(text) {
    if (!('speechSynthesis' in window)) return;
    speechSynthesis.cancel(); // the latest instruction wins over queued ones
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    const voice = pickEnglishVoice();
    if (voice) utterance.voice = voice;
    utterance.rate = 0.9;
    speechSynthesis.speak(utterance);
}

function playBeep(frequency = 440, duration = 200) {
    const ctx = getAudioCtx();
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;

    gainNode.gain.setValueAtTime(0.4, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration / 1000);

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.start();
    oscillator.stop(ctx.currentTime + duration / 1000 + 0.05);
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
        // Warm up the async voice list so an English voice is ready by the first speak()
        if ('speechSynthesis' in window) {
            speechSynthesis.getVoices();
            speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
        }

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
    
    // Timeout: 30s to get seated and calibrated, then 30s for the walk itself
    // (TUG time is measured from the "stand up" instruction, so a slow start
    // must not eat into the walking budget).
    const timeoutBase = goSignalAt || startTime;
    if (Date.now() - timeoutBase > 30000 && currentState !== STATES.DONE) {
        playBeep(300, 500);
        speak("Time is up. Recording stopped.");
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
            const leftKnee = keypoints[13];
            const rightKnee = keypoints[14];

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

                kneesVisible = leftKnee.score > 0.3 && rightKnee.score > 0.3;
                if (kneesVisible) {
                    const currentKneesY = (leftKnee.y + rightKnee.y) / 2;
                    smoothedKneesY = smoothedKneesY === 0
                        ? currentKneesY
                        : EMA_ALPHA * currentKneesY + (1 - EMA_ALPHA) * smoothedKneesY;
                }

                const timeSinceStart = Date.now() - startTime;
                const distFromStart = Math.abs(smoothedHipsX - calibrationData.startX);
                // Assume threshold is 15% of torso length for standing detection
                const standThreshold = calibrationData.baseHipsY - (calibrationData.baseTorsoLength * 0.15); 

                switch (currentState) {
                    case STATES.PRE_CALIBRATION:
                        // Give 4 seconds to sit down, and only calibrate once the pose
                        // actually reads as seated (unknown knees degrade to time-only).
                        if (timeSinceStart > 4000) {
                            if (isSeated() === false) {
                                if (Date.now() - lastSitPromptAt > 6000) {
                                    lastSitPromptAt = Date.now();
                                    playBeep(440, 150);
                                    speak("Please sit on the chair and stay still.");
                                }
                            } else {
                                setState(STATES.CALIBRATING);
                                calibrationStartedAt = Date.now();
                                calibrationData.samples = 0;
                                calibrationData.baseHipsY = 0;
                                calibrationData.startX = 0;
                                calibrationData.baseTorsoLength = 0;
                            }
                        }
                        break;

                    case STATES.CALIBRATING:
                        // Standing up mid-calibration would poison the seated baseline
                        if (isSeated() === false) {
                            setState(STATES.PRE_CALIBRATION);
                            break;
                        }

                        calibrationData.baseHipsY += smoothedHipsY;
                        calibrationData.startX += smoothedHipsX;
                        calibrationData.baseTorsoLength += smoothedTorso;
                        calibrationData.samples++;

                        // 1.5s of calibration; require enough valid frames so the
                        // baseline is never a division by zero when tracking drops out
                        if (Date.now() - calibrationStartedAt > 1500 && calibrationData.samples >= 10) {
                            calibrationData.baseHipsY /= calibrationData.samples;
                            calibrationData.startX /= calibrationData.samples;
                            calibrationData.baseTorsoLength /= calibrationData.samples;

                            setState(STATES.WAITING_TO_STAND);
                            goSignalAt = Date.now(); // TUG timing starts here
                            playBeep(600, 300); // "Go" signal
                            speak("Stand up and walk three meters forward.");
                        }
                        break;

                    case STATES.WAITING_TO_STAND:
                        // Y coordinate decreases when standing up (moves up on screen)
                        if (smoothedHipsY < standThreshold) {
                            setState(STATES.WALKING_AWAY);
                            maxDistance = 0;
                            playBeep(700, 200); // Signal movement start
                            speak("Walk forward.");
                        }
                        break;

                    case STATES.WALKING_AWAY:
                        if (distFromStart > maxDistance) {
                            maxDistance = distFromStart;
                        }
                        // Turn detection: max distance reached (at least 30px) and now going back
                        if (maxDistance > 30 && distFromStart < maxDistance - 15) {
                            setState(STATES.WALKING_BACK);
                            playBeep(500, 200);
                            speak("Turn around and walk back to the chair.");
                        }
                        break;

                    case STATES.WALKING_BACK:
                        // Returned close to start X, and hips lowered again
                        if (distFromStart < 50 && smoothedHipsY > standThreshold) {
                            setState(STATES.SITTING_DOWN);
                            stateTimer = Date.now();
                            // Hovering over the chair re-enters this state every few
                            // frames — don't repeat the cue each time
                            if (Date.now() - lastSitDownCueAt > 4000) {
                                lastSitDownCueAt = Date.now();
                                playBeep(700, 200);
                                speak("Sit down on the chair.");
                            }
                        }
                        break;

                    case STATES.SITTING_DOWN:
                        if (smoothedHipsY < standThreshold || isSeated() === false) {
                            // False alarm: stood up again / hovering at the chair without sitting
                            setState(STATES.WALKING_BACK);
                        } else if (Date.now() - stateTimer > 1000) {
                            setState(STATES.DONE);
                            playBeep(800, 200); // Signal end
                            setTimeout(() => playBeep(1000, 350), 250);
                            speak("Test completed. Well done.");
                            endBtn.click(); // Auto-stop
                        }
                        break;
                }
            }

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
    // Unlocks Audio/Speech context on user interaction and speaks first instruction
    getAudioCtx();
    playBeep(440, 150);
    speak("Please sit on the chair and stay still.");

    isRecording = true;
    recordedData = [];
    startTime = Date.now();

    // Reset FSM
    setState(STATES.PRE_CALIBRATION);
    stateLabel.classList.remove('hidden');
    calibrationData = { baseHipsY: 0, baseTorsoLength: 0, startX: 0, samples: 0 };
    calibrationStartedAt = 0;
    goSignalAt = 0;
    lastSitPromptAt = Date.now(); // the sit instruction was just spoken above
    lastSitDownCueAt = 0;
    smoothedHipsX = 0;
    smoothedHipsY = 0;
    smoothedTorso = 0;
    smoothedKneesY = 0;
    kneesVisible = false;
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
    if (currentState !== STATES.DONE) {
        // Manual stop mid-test: the last FSM state is stale, hide it
        stateLabel.classList.add('hidden');
    }
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
