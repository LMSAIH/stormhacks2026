import {
  INPUT_SIZE,
  PATCH_SIZE,
  applyAffine,
  estimateSimilarity,
  patchOrigin,
} from "@/lib/lipreading/crop"
import type { Keypoints, Point } from "@/lib/lipreading/types"

const MOUTH_COLOR = "oklch(0.72 0.19 150)"
const POINT_COLOR = "oklch(0.97 0 0)"

/**
 * Draw the 4 keypoints and the mouth region the model sees onto the overlay canvas. Called from
 * the capture loop every frame (not a React render), onto whatever `<canvas>` the camera panel
 * hands out as `overlayRef` — the panel itself stays a plain video + canvas. `frameWidth/Height`
 * are the video's pixel size; the mapping matches the video's `object-cover` fit, so it holds for
 * any panel aspect ratio. The canvas sits in the same mirrored box as the video, so drawing in
 * un-mirrored frame coordinates lines up.
 */
export function drawFaceOverlay(
  canvas: HTMLCanvasElement | null,
  frameWidth: number,
  frameHeight: number,
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
  if (!keypoints || frameWidth === 0 || frameHeight === 0) return

  const scale = Math.max(width / frameWidth, height / frameHeight)
  const offsetX = (width - frameWidth * scale) / 2
  const offsetY = (height - frameHeight * scale) / 2
  const toCanvas = ([x, y]: Point): Point => [
    offsetX + x * scale,
    offsetY + y * scale,
  ]

  const box = modelInputCorners(keypoints)
  if (box) {
    ctx.beginPath()
    box.forEach((corner, i) => {
      const [x, y] = toCanvas(corner)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.lineWidth = 1.5 * dpr
    ctx.setLineDash([4 * dpr, 3 * dpr])
    ctx.strokeStyle = MOUTH_COLOR
    ctx.stroke()
    ctx.setLineDash([])
  }

  ctx.lineWidth = dpr
  ctx.strokeStyle = "rgb(0 0 0 / 0.45)"
  keypoints.forEach((point, i) => {
    const [x, y] = toCanvas(point)
    ctx.beginPath()
    ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2)
    ctx.fillStyle = i === 3 ? MOUTH_COLOR : POINT_COLOR
    ctx.fill()
    ctx.stroke()
  })
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
