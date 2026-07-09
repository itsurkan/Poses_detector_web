// @ts-check
// End-to-end tests for the TUG test FSM in movenet.html.
//
// The real camera and MoveNet model are stubbed before page scripts run:
// - getUserMedia returns a fake canvas stream,
// - tf / poseDetection are replaced, the detector returns window.__pose,
// - CDN scripts are aborted (stubs make them unnecessary), so tests run offline.
//
// Playwright's fake clock drives requestAnimationFrame and Date.now, so a full
// 15-second test scenario runs in milliseconds of real time. Tests move the
// "person" by assigning window.__pose and then advancing the clock.
const { test, expect } = require('@playwright/test');

// Keypoint indices: 5/6 shoulders, 11/12 hips, 13/14 knees.
// Geometry (baseline torso = 100px): seated knees ≈ hip level, standing knees
// a thigh-length below the hips. Stand threshold ends up at 400 - 15 = 385.
const POSES = {
    sitting:        { x: 320, shY: 300, hipY: 400, kneeY: 410 },
    standing:       { x: 320, shY: 240, hipY: 340, kneeY: 440 },
    walkedAway:     { x: 200, shY: 240, hipY: 340, kneeY: 440 },
    turnedBack:     { x: 240, shY: 240, hipY: 340, kneeY: 440 },
    atChairStanding:{ x: 320, shY: 240, hipY: 340, kneeY: 440 },
    // Hips low enough to pass the old threshold check, but knees still a
    // thigh-length below the hips — crouching/hovering, NOT seated.
    atChairCrouch:  { x: 320, shY: 300, hipY: 400, kneeY: 445 },
};

const PHRASES = {
    sit: 'Please sit on the chair and stay still.',
    go: 'Stand up and walk three meters forward.',
    walk: 'Walk forward.',
    turn: 'Turn around and walk back to the chair.',
    sitDown: 'Sit down on the chair.',
    done: 'Test completed. Well done.',
    timeout: 'Time is up. Recording stopped.',
};

async function setupPage(page) {
    // The stubs make the CDN payloads unnecessary — drop them for speed/offline runs
    await page.route('https://cdn.jsdelivr.net/**', route => route.abort());
    await page.route('https://fonts.googleapis.com/**', route => route.abort());
    await page.route('https://fonts.gstatic.com/**', route => route.abort());

    await page.addInitScript(() => {
        // Fake camera: a repainting canvas keeps the captured stream alive
        const cam = document.createElement('canvas');
        cam.width = 640;
        cam.height = 480;
        const camCtx = cam.getContext('2d');
        camCtx.fillStyle = '#222';
        camCtx.fillRect(0, 0, 640, 480);
        setInterval(() => {
            camCtx.fillStyle = '#' + (Math.random() * 0xFFFFFF | 0).toString(16).padStart(6, '0');
            camCtx.fillRect(0, 0, 20, 20);
        }, 100);
        navigator.mediaDevices.getUserMedia = async () => cam.captureStream(30);

        // Speech recorder
        window.__spoken = [];
        window.speechSynthesis.speak = (u) => {
            window.__spoken.push({ text: u.text, lang: u.lang });
        };

        // Detector stub: returns whatever pose the test currently dictates
        window.__pose = null;
        function poseKeypoints(p) {
            const kp = [];
            for (let i = 0; i < 17; i++) kp.push({ x: p.x, y: p.shY - 40, score: 0.9 });
            kp[5] = { x: p.x - 30, y: p.shY, score: 0.9 };
            kp[6] = { x: p.x + 30, y: p.shY, score: 0.9 };
            kp[11] = { x: p.x - 20, y: p.hipY, score: 0.9 };
            kp[12] = { x: p.x + 20, y: p.hipY, score: 0.9 };
            kp[13] = { x: p.x - 20, y: p.kneeY, score: 0.9 };
            kp[14] = { x: p.x + 20, y: p.kneeY, score: 0.9 };
            return kp;
        }
        window.tf = { setBackend: async () => true, ready: async () => true };
        window.poseDetection = {
            movenet: { modelType: { THUNDER: 'thunder' } },
            SupportedModels: { MoveNet: 'MoveNet' },
            createDetector: async () => ({
                estimatePoses: async () => window.__pose ? [{ keypoints: poseKeypoints(window.__pose) }] : [],
            }),
        };
    });

    await page.goto('/movenet.html');
    await expect(page.locator('#startBtn')).toBeEnabled();

    // Beep recorder: playBeep is a global function declaration, wrap it now.
    // Beeps only ever fire after the Start click, so wrapping post-load is safe.
    await page.evaluate(() => {
        window.__beeps = [];
        const realBeep = window.playBeep;
        window.playBeep = (f, d) => { window.__beeps.push(f); realBeep(f, d); };
    });

    // Fake clock from here on: installed after init so camera/model setup ran
    // under the real clock, but before Start so all FSM timing is driven by tests.
    await page.clock.install();
}

const setPose = (page, pose) => page.evaluate(p => { window.__pose = p; }, pose);
const spoken = (page) => page.evaluate(() => window.__spoken.map(s => s.text));
const beeps = (page) => page.evaluate(() => window.__beeps);
const state = (page) => page.locator('#stateLabel');

async function runFor(page, ms) {
    // Advance in rAF-sized steps so the processFrame loop and EMA smoothing tick
    for (let t = 0; t < ms; t += 1000) {
        await page.clock.runFor(Math.min(1000, ms - t));
    }
}

/** Drive the FSM from Start click up to the "go" signal (person seated). */
async function reachGoSignal(page) {
    await setPose(page, POSES.sitting);
    await page.locator('#startBtn').click();
    await runFor(page, 4300);  // PRE_CALIBRATION window
    await expect(state(page)).toHaveText('КАЛІБРУВАННЯ');
    await runFor(page, 1700);  // calibration
    await expect(state(page)).toHaveText('ОЧІКУВАННЯ РУХУ');
}

/** Drive the FSM from the go signal to the point of returning to the chair, still standing. */
async function walkOutAndReturn(page) {
    await setPose(page, POSES.standing);
    await runFor(page, 1000);
    await expect(state(page)).toHaveText('ВІДДАЛЕННЯ');
    await setPose(page, POSES.walkedAway);
    await runFor(page, 1000);
    await setPose(page, POSES.turnedBack);
    await runFor(page, 1000);
    await expect(state(page)).toHaveText('ПОВЕРНЕННЯ');
    await setPose(page, POSES.atChairStanding);
    await runFor(page, 1000);
}

test.describe('TUG test flow', () => {
    test.beforeEach(async ({ page }) => setupPage(page));

    test('full test: every phase gets a voice instruction and a beep, ends with auto-save', async ({ page }) => {
        await reachGoSignal(page);
        await walkOutAndReturn(page);

        const downloadPromise = page.waitForEvent('download');
        await setPose(page, POSES.sitting);
        await runFor(page, 500);
        await expect(state(page)).toHaveText('СІДАННЯ');
        await runFor(page, 1500);
        await expect(state(page)).toHaveText('ЗАВЕРШЕНО');

        expect(await spoken(page)).toEqual([
            PHRASES.sit, PHRASES.go, PHRASES.walk, PHRASES.turn, PHRASES.sitDown, PHRASES.done,
        ]);
        expect(await beeps(page)).toEqual([440, 600, 700, 500, 700, 800, 1000]);

        // All speech is English — Ukrainian voices are not available in browsers
        const langs = await page.evaluate(() => [...new Set(window.__spoken.map(s => s.lang))]);
        expect(langs).toEqual(['en-US']);

        // Auto-stop saved the recording and reset the controls
        const download = await downloadPromise;
        expect(download.suggestedFilename()).toMatch(/^pose_data_movenet_\d+\.json$/);
        await expect(page.locator('#startBtn')).toBeEnabled();
        await expect(page.locator('#endBtn')).toBeDisabled();
        await expect(page.locator('#statusText')).toHaveText('Ready');
        await expect(state(page)).toBeVisible();
    });

    test('start gate: calibration waits until the person actually sits, then re-prompts', async ({ page }) => {
        await setPose(page, POSES.standing); // person stands instead of sitting
        await page.locator('#startBtn').click();

        await runFor(page, 4500);
        await expect(state(page)).toHaveText('ПІДГОТОВКА'); // not calibrating on a standing person

        await runFor(page, 7000); // past the 6s re-prompt throttle
        const phrases = await spoken(page);
        expect(phrases.filter(t => t === PHRASES.sit).length).toBeGreaterThanOrEqual(2); // re-prompted
        expect(phrases).not.toContain(PHRASES.go);

        await setPose(page, POSES.sitting); // now they sit
        await runFor(page, 1500);
        await expect(state(page)).toHaveText('КАЛІБРУВАННЯ');
        await runFor(page, 1700);
        await expect(state(page)).toHaveText('ОЧІКУВАННЯ РУХУ');
        expect((await spoken(page)).filter(t => t === PHRASES.go)).toEqual([PHRASES.go]); // go given once, only after sitting
    });

    test('end gate: hovering at the chair without sitting does not complete the test', async ({ page }) => {
        await reachGoSignal(page);
        await walkOutAndReturn(page);

        // Crouch at the chair: hips low (passes the old threshold), knees say "not seated"
        await setPose(page, POSES.atChairCrouch);
        await runFor(page, 3000);
        expect(await spoken(page)).not.toContain(PHRASES.done);
        await expect(state(page)).not.toHaveText('ЗАВЕРШЕНО');

        // Actually sitting down completes it
        await setPose(page, POSES.sitting);
        await runFor(page, 2000);
        await expect(state(page)).toHaveText('ЗАВЕРШЕНО');
        expect(await spoken(page)).toContain(PHRASES.done);
    });

    test('timeout: 30s after the go signal without finishing stops the recording', async ({ page }) => {
        await reachGoSignal(page);
        // Person keeps sitting and never stands up
        await runFor(page, 31_000);
        expect(await spoken(page)).toContain(PHRASES.timeout);
        expect(await beeps(page)).toContain(300);
        await expect(page.locator('#statusText')).toHaveText('Ready');
        await expect(state(page)).toBeHidden(); // stale mid-test state is not shown
    });

    test('timeout: never sitting down at all also stops after 30s', async ({ page }) => {
        await setPose(page, POSES.standing);
        await page.locator('#startBtn').click();
        await runFor(page, 31_000);
        expect(await spoken(page)).toContain(PHRASES.timeout);
        expect(await spoken(page)).not.toContain(PHRASES.go);
    });

    test('manual stop mid-test saves the recording and hides the stale state', async ({ page }) => {
        await setPose(page, POSES.sitting);
        await page.locator('#startBtn').click();
        await runFor(page, 4500);

        const downloadPromise = page.waitForEvent('download');
        await page.locator('#endBtn').click();
        const download = await downloadPromise;
        expect(download.suggestedFilename()).toMatch(/^pose_data_movenet_\d+\.json$/);
        await expect(state(page)).toBeHidden();
        await expect(page.locator('#startBtn')).toBeEnabled();
        expect(await spoken(page)).not.toContain(PHRASES.done);
    });

    test('restart: a second run works and repeats all instructions', async ({ page }) => {
        await reachGoSignal(page);
        await page.locator('#endBtn').click(); // abort first run

        await setPose(page, POSES.sitting);
        await page.locator('#startBtn').click();
        await runFor(page, 4300);
        await expect(state(page)).toHaveText('КАЛІБРУВАННЯ');
        await runFor(page, 1700);
        await expect(state(page)).toHaveText('ОЧІКУВАННЯ РУХУ');
        expect((await spoken(page)).filter(t => t === PHRASES.go).length).toBe(2);
    });
});

test.describe('fullscreen layout', () => {
    test.beforeEach(async ({ page }) => setupPage(page));

    for (const [name, viewport] of Object.entries({
        desktop: { width: 1280, height: 800 },
        mobile: { width: 375, height: 812 },
    })) {
        test(`camera fills the ${name} viewport and controls sit on top of it`, async ({ page }) => {
            await page.setViewportSize(viewport);

            const stage = await page.locator('.camera-stage').boundingBox();
            expect(stage).toEqual(expect.objectContaining({
                x: 0, y: 0, width: viewport.width, height: viewport.height,
            }));

            const video = await page.locator('#webcam').boundingBox();
            expect(video.width).toBe(viewport.width);
            expect(video.height).toBe(viewport.height);

            // Buttons render inside the viewport, over the video, and receive clicks
            const btn = await page.locator('#startBtn').boundingBox();
            expect(btn.y + btn.height).toBeLessThanOrEqual(viewport.height);
            const clickable = await page.evaluate(() => {
                const b = document.getElementById('startBtn');
                const r = b.getBoundingClientRect();
                return document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2) === b;
            });
            expect(clickable).toBe(true);
        });
    }
});
