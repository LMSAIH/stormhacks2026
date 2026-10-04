import { useEffect, useRef } from "react"

import { ACCENT_COLORS } from "@/lib/palette"

export interface StrandsProps {
  /** Number of strands. */
  count?: number
  /** How wavy each strand is (undulation frequency). */
  waviness?: number
  /** Fill opacity. */
  intensity?: number
  /** How sharply strands taper to points at the ends (higher = pointier). */
  taper?: number
  /** Overall thickness multiplier. */
  scale?: number
  /** Flow speed. */
  speed?: number
  /** Vertical wave size multiplier. */
  amplitude?: number
  className?: string
}

/** Flowing, tapered "strands" ribbons on a canvas — inspired by reactbits.dev/animations/strands. */
export function Strands({
  count = 4,
  waviness = 1.8,
  intensity = 0.4,
  taper = 2.5,
  scale = 2.2,
  speed = 0.1,
  amplitude = 1.9,
  className,
}: StrandsProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const parent = canvas?.parentElement
    const ctx = canvas?.getContext("2d")
    if (!canvas || !parent || !ctx) return

    const colors = Array.from(
      { length: count },
      (_, i) => ACCENT_COLORS[i % ACCENT_COLORS.length]
    )
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    const isDark = () => document.documentElement.classList.contains("dark")

    let raf = 0
    let t = 0
    let w = 0
    let h = 0

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = parent.clientWidth
      h = parent.clientHeight
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const TAU = Math.PI * 2
    const STEPS = 160

    const yAt = (xn: number, baseY: number, baseAmp: number, phase: number) =>
      baseY +
      baseAmp *
        (Math.sin(xn * TAU * waviness + t + phase) +
          0.5 * Math.sin(xn * TAU * waviness * 1.9 + t * 1.3 + phase * 1.7))

    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      const baseAmp = h * 0.045 * amplitude
      const maxHalf = 7 * scale
      const alpha = intensity * (isDark() ? 1 : 0.8)

      for (let s = 0; s < count; s++) {
        const baseY = (h * (s + 1)) / (count + 1)
        const phase = s * 1.7
        ctx.beginPath()
        for (let i = 0; i <= STEPS; i++) {
          const xn = i / STEPS
          const y = yAt(xn, baseY, baseAmp, phase)
          const hw = maxHalf * Math.pow(Math.sin(Math.PI * xn), taper)
          const px = xn * w
          if (i === 0) ctx.moveTo(px, y - hw)
          else ctx.lineTo(px, y - hw)
        }
        for (let i = STEPS; i >= 0; i--) {
          const xn = i / STEPS
          const y = yAt(xn, baseY, baseAmp, phase)
          const hw = maxHalf * Math.pow(Math.sin(Math.PI * xn), taper)
          ctx.lineTo(xn * w, y + hw)
        }
        ctx.closePath()
        ctx.fillStyle = withAlpha(colors[s], alpha)
        ctx.fill()
      }
    }

    const loop = () => {
      t += speed * 0.04
      draw()
      raf = requestAnimationFrame(loop)
    }

    resize()
    const ro = new ResizeObserver(() => {
      resize()
      if (reduce) draw()
    })
    ro.observe(parent)

    if (reduce) draw()
    else raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [count, waviness, intensity, taper, scale, speed, amplitude])

  return <canvas ref={canvasRef} className={className} aria-hidden />
}

/** "#rrggbb" + alpha → "rgba(...)". */
function withAlpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`
}
