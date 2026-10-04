// App-level regression: play eval20.y4m (30 s still lead, 20 real-face clips, 1.5 s still after
// each) through /app in MODE, collect the finished lines, and write them for WER scoring.
//   MODE=normal node e2e_eval.mjs   (BASE default http://localhost:5300)
//   needs: PLAYWRIGHT_CORE=<path to playwright-core/index.mjs> CHROME=<chromium binary>
// Writes eval_app_<TAG>.json (lines + cut summary) and trace_<TAG>.json (the app's dev trace:
// tracker results, cuts, reads; `cuts.py` maps it onto the clips).
import { writeFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? "playwright-core")

const BASE = process.env.BASE ?? "http://localhost:5300"
const OUT = process.env.OUT ?? fileURLToPath(new URL("../../artifacts/app_eval", import.meta.url))
const MODE = process.env.MODE ?? "normal"
const TAG = process.env.TAG ?? MODE
const WATCH_S = Number(process.env.WATCH_S ?? 135) // one pass of the 120 s video + the last read
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROME,
  args: [
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${OUT}/eval20.y4m`,
  ],
})
const context = await browser.newContext({ viewport: { width: 1280, height: 800 } })
await context.addInitScript((m) => {
  try {
    localStorage.setItem("lipread.mode", m)
  } catch {}
}, MODE)
const page = await context.newPage()
page.on("pageerror", (e) => console.log("pageerror:", e.message.slice(0, 200)))
const t0 = Date.now()
await page.goto(`${BASE}/app`)
// performance.now() at the fake camera's first frame: maps the trace onto the video's clips.
let origin = null
while (origin === null && Date.now() - t0 < 30_000) {
  origin = await page.evaluate(() => {
    const v = document.querySelector("video")
    return v && v.currentTime > 0 ? performance.now() - v.currentTime * 1000 : null
  })
  if (origin === null) await page.waitForTimeout(100)
}
await page.waitForTimeout(Math.max(0, WATCH_S * 1000 - (Date.now() - t0)))
const { lines, trace } = await page.evaluate(() => {
  // Finished lines carry data-lip-line (self-transcript.tsx); older layouts: spans in the "You" box.
  let lines = [...document.querySelectorAll("[data-lip-line]")].map((p) => p.textContent.trim())
  if (!lines.length) {
    const box = [...document.querySelectorAll("span")].find((s) => s.textContent === "You")?.closest(".rounded-xl")
    lines = box ? [...box.querySelectorAll("p > span:not(.italic)")].map((s) => s.textContent.replace(/ · $/, "")) : []
  }
  return { lines, trace: window.__lipTrace ?? [] }
})
await browser.close()

const locks = trace
  .filter((e) => e.kind === "lock")
  .map((l) => ({ reason: l.reason, s: +(((l.endTms ?? l.tMs) - l.startTms) / 1000).toFixed(2) }))
const finals = trace
  .filter((e) => e.kind === "final")
  .map((f) => ({ read: f.read, shown: f.shown, snapped: f.snapped, blocked: f.blocked }))
const drops = trace.filter((e) => e.kind === "drop").map(({ at, kind, ...rest }) => rest)
// Lip tracking health: results per second and how far they lag the camera.
const results = trace.filter((e) => e.kind === "result")
const median = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : null)
const span = results.length > 1 ? (results.at(-1).tMs - results[0].tMs) / 1000 : 0
const tracker = {
  hz: span ? +(results.length / span).toFixed(1) : null,
  lagMs: median(results.map((e) => Math.round(e.at - e.tMs))),
}
const result = { mode: TAG, tracker, origin, lines, locks, finals, drops }
writeFileSync(`${OUT}/eval_app_${TAG}.json`, JSON.stringify(result, null, 1))
writeFileSync(`${OUT}/trace_${TAG}.json`, JSON.stringify(trace))
console.log(MODE, "lines:", lines.length, "locks:", locks.length, "tracker:", tracker)
if (!lines.length && finals.length) console.log("warning: the app read lines but none were found on the page")
