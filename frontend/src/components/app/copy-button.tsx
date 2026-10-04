import { useRef, useState } from "react"
import { Check, Copy } from "lucide-react"

import { Button } from "@/components/ui/button"

interface CopyButtonProps {
  /** Produces the text to copy, evaluated on click so it's always current. */
  getText: () => string
  /** Optional visible label; omit for an icon-only button. */
  label?: string
  title?: string
  size?: "sm" | "icon-sm" | "icon-xs" | "default" | "lg"
  variant?: "ghost" | "outline" | "default" | "secondary"
  className?: string
}

/** Copies text to the clipboard with a brief "copied" confirmation. */
export function CopyButton({
  getText,
  label,
  title = "Copy transcript",
  size = "icon-sm",
  variant = "ghost",
  className,
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  const copy = async () => {
    const text = getText().trim()
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // Fallback for non-secure contexts / older browsers.
      const ta = document.createElement("textarea")
      ta.value = text
      ta.style.position = "fixed"
      ta.style.opacity = "0"
      document.body.appendChild(ta)
      ta.select()
      try {
        document.execCommand("copy")
      } catch {
        /* nothing more we can do */
      }
      ta.remove()
    }
    setCopied(true)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <Button
      size={size}
      variant={variant}
      onClick={copy}
      aria-label={title}
      title={copied ? "Copied!" : title}
      className={className}
    >
      {copied ? <Check /> : <Copy />}
      {label && (copied ? "Copied" : label)}
    </Button>
  )
}
