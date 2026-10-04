import { Link } from "react-router-dom"
import { ArrowRight } from "lucide-react"
import { cn } from "cn"

import { buttonVariants } from "@/components/ui/button"
import SoftAurora from "@/components/ui/soft-aurora"

/** Landing is always dark — Soft Aurora reads best on a dark background. */
export function LandingPage() {
  return (
    <div className="dark relative flex h-svh flex-col overflow-hidden bg-background text-foreground">
      {/* Aurora background */}
      <div className="pointer-events-none absolute inset-0">
        <SoftAurora
          lightMode={false}
          color1="#ffffff"
          color2="#7c3aed"
          speed={0.5}
          scale={1.5}
          brightness={1}
          enableMouseInteraction={false}
        />
      </div>

      {/* Legibility scrim behind the headline */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          backdropFilter: "blur(10px)",
          WebkitBackdropFilter: "blur(10px)",
          backgroundColor: "rgba(10,10,12,0.28)",
          maskImage:
            "radial-gradient(62% 48% at 50% 30%, black 40%, transparent 74%)",
          WebkitMaskImage:
            "radial-gradient(62% 48% at 50% 30%, black 40%, transparent 74%)",
        }}
      />
      {/* Top scrim for the nav */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-28"
        style={{
          background: "linear-gradient(to bottom, rgba(10,10,12,0.6), transparent)",
        }}
      />
      {/* Bottom scrim for the explainer text */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 bottom-0 h-44"
        style={{
          background: "linear-gradient(to top, rgba(10,10,12,0.65), transparent)",
        }}
      />

      {/* Nav */}
      <header className="relative z-10 flex items-center justify-between px-8 pt-9 pb-5">
        <Link to="/" className="flex items-center">
          <span className="text-lg font-semibold italic tracking-tight">heard</span>
        </Link>
        <Link
          to="/login"
          className={cn(buttonVariants({ variant: "ghost", size: "sm" }), "h-9 px-4")}
        >
          Sign in
        </Link>
      </header>

      {/* Hero */}
      <main className="relative z-10 flex flex-1 items-start justify-center px-6 pt-[7vh]">
        <div className="mx-auto max-w-xl text-center">
          <h1 className="text-[2.1rem] leading-[1.1] font-semibold tracking-tight sm:text-5xl">
            Everyone deserves to be <span className="italic">heard</span>.
          </h1>
          <p className="mx-auto mt-5 max-w-md text-sm leading-relaxed text-muted-foreground sm:text-base">
            Your lips become your voice. Their words become your captions.
          </p>
          <div className="mt-8 flex items-center justify-center">
            <Link
              to="/app"
              className={cn(buttonVariants({ size: "lg" }), "h-11 px-6 text-sm")}
            >
              Start speaking
              <ArrowRight />
            </Link>
          </div>
        </div>
      </main>

      {/* Explainer, centered at the bottom */}
      <div className="relative z-10 mx-auto max-w-lg space-y-3 px-6 pb-10 text-center">
        <p className="text-sm leading-relaxed text-muted-foreground">
          Mouth your words and heard speaks for you, then hear everyone back as
          live, speaker-labeled captions. On-device, in real time.
        </p>
        <Link
          to="/privacy"
          className="inline-block text-xs text-muted-foreground/70 underline underline-offset-4 transition-colors hover:text-foreground"
        >
          Privacy
        </Link>
      </div>
    </div>
  )
}

export default LandingPage
