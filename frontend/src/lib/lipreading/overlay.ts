import {
  INPUT_SIZE,
  PATCH_SIZE,
  applyAffine,
  estimateSimilarity,
  patchOrigin,
} from "@/lib/lipreading/crop"
import type { Keypoints, Point } from "@/lib/lipreading/types"

/** The lip dots' colour from the teammate's overlay; the model-crop box shares it. */
const MOUTH_COLOR = "oklch(0.72 0.19 150)"
/** Lip dot radius in CSS px (his value). */
const LIP_DOT_RADIUS = 1.5

/** A landmark in the video frame: `x` / `y` are 0..1 of its width / height (MediaPipe's unit). */
export interface NormalizedPoint {
  readonly x: number
  readonly y: number
}

/**
 * Draw the lip tracking onto the overlay canvas: the lip-contour dots from `LipLandmarker` — the
 * teammate's overlay, same colour and size — are the primary element and are always drawn when
 * present; the mouth region the model sees (from the BlazeFace `keypoints`) is added underneath as
 * a faint dashed box. Called from the capture loop every frame (not a React render), onto
 * whatever `<canvas>` the camera panel hands out as `overlayRef` — the panel itself stays a plain
 * video + canvas. `frameWidth/Height` are the video's pixel size; the mapping matches the video's
 * `object-cover` fit, so the dots stay on the lips for any panel aspect ratio (for a panel with
 * the camera's own aspect ratio it reduces to `x * panelWidth`, as in his version). The canvas
 * sits in the same mirrored box as the video, so drawing in un-mirrored frame coordinates lines
 * up.
 */
export function drawFaceOverlay(
  canvas: HTMLCanvasElement | null,
  frameWidth: number,
  frameHeight: number,
  lipPoints: readonly NormalizedPoint[],
  keypoints: Keypoints | null
): void {
  if (!canvas) return
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const cssWidth = canvas.clientWidth
  const cssHeight = canvas.clientHeight
  if (cssWidth === 0 || cssHeight === 0) return
  const dpr = window.devicePixelRatio || 1
  const width = Math.round(cssWidth * dpr)
  const height = Math.round(cssHeight * dpr)
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
  ctx.clearRect(0, 0, width, height)
  if (frameWidth === 0 || frameHeight === 0) return

  const scale = Math.max(width / frameWidth, height / frameHeight)
  const offsetX = (width - frameWidth * scale) / 2
  const offsetY = (height - frameHeight * scale) / 2
  const toCanvas = ([x, y]: Point): Point => [
    offsetX + x * scale,
    offsetY + y * scale,
  ]

  // Underneath: what the model sees, faint so the lip dots stay the focus.
  const box = keypoints && modelInputCorners(keypoints)
  if (box) {
    ctx.beginPath()
    box.forEach((corner, i) => {
      const [x, y] = toCanvas(corner)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.lineWidth = dpr
    ctx.setLineDash([4 * dpr, 3 * dpr])
    ctx.strokeStyle = MOUTH_COLOR
    ctx.globalAlpha = 0.55
    ctx.stroke()
    ctx.globalAlpha = 1
    ctx.setLineDash([])
  }

  // The lip dots, in the teammate's colour and radius: one filled circle per lip landmark.
  ctx.fillStyle = MOUTH_COLOR
  for (const p of lipPoints) {
    const [x, y] = toCanvas([p.x * frameWidth, p.y * frameHeight])
    ctx.beginPath()
    ctx.arc(x, y, LIP_DOT_RADIUS * dpr, 0, Math.PI * 2)
    ctx.fill()
  }
}

/**
 * Source-pixel corners of the 88² model input for one frame's raw keypoints, using the crop
 * pipeline's own geometry (similarity onto the mean face → 96² patch origin → centre 88²) mapped
 * back through the inverse transform. Approximate only because the real crop smooths keypoints
 * over ±6 frames first.
 */
function modelInputCorners(kps: Keypoints): Point[] | null {
  const m = estimateSimilarity(kps)
  const [a, b, c, d, e, f] = m
  const det = a * e - b * d
  if (!m.every(Number.isFinite) || det === 0) return null
  const [cx, cy] = applyAffine(m, kps[3])
  const [px, py] = patchOrigin(cx, cy)
  const margin = (PATCH_SIZE - INPUT_SIZE) / 2
  const x0 = px + margin
  const y0 = py + margin

  const toSource = (u: number, v: number): Point => [
    (e * (u - c) - b * (v - f)) / det,
    (a * (v - f) - d * (u - c)) / det,
  ]
  return [
    toSource(x0, y0),
    toSource(x0 + INPUT_SIZE, y0),
    toSource(x0 + INPUT_SIZE, y0 + INPUT_SIZE),
    toSource(x0, y0 + INPUT_SIZE),
  ]
}
