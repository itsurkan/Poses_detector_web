// @ts-check
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
    testDir: './tests',
    timeout: 60_000,
    fullyParallel: true,
    reporter: [['list']],
    use: {
        baseURL: 'http://127.0.0.1:4173',
    },
    webServer: {
        command: 'python3 -m http.server 4173 --bind 127.0.0.1',
        url: 'http://127.0.0.1:4173/movenet.html',
        reuseExistingServer: true,
    },
});
