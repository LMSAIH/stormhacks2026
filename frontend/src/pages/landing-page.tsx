import { Link } from "react-router-dom"
import { ArrowRight, AudioLines } from "lucide-react"
import { cn } from "cn"

import { buttonVariants } from "@/components/ui/button"
import { ThemeToggle } from "@/components/app/theme-toggle"
import { Strands } from "@/components/app/strands"
import { ACCENT_COLORS } from "@/lib/palette"

/** Flowing strands behind the hero, softly masked so text stays readable. */
function HearingThreads() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden"
      style={{
        maskImage:
          "radial-gradient(120% 85% at 50% 45%, transparent 24%, black 72%)",
        WebkitMaskImage:
          "radial-gradient(120% 85% at 50% 45%, transparent 24%, black 72%)",
      }}
    >
      <Strands
        className="h-full w-full"
        count={4}
        waviness={1.8}
        intensity={0.4}
        taper={2.5}
        scale={2.2}
        speed={0.1}
        amplitude={1.9}
      />
    </div>
  )
}

export function LandingPage() {
  return (
    <div className="relative flex min-h-svh flex-col overflow-x-hidden bg-background text-foreground">
      {/* Nav */}
      <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-5">
        <Wordmark />
        <nav className="flex items-center gap-2">
          <Link
            to="/login"
            className={cn(
              buttonVariants({ variant: "ghost", size: "sm" }),
              "hidden sm:inline-flex"
            )}
          >
            Sign in
          </Link>
          <Link to="/app" className={buttonVariants({ size: "sm" })}>
            Open heard
            <ArrowRight />
          </Link>
          <ThemeToggle />
        </nav>
      </header>

      {/* Hero */}
      <section className="relative flex flex-1 items-center">
        <HearingThreads />
        <div className="relative z-10 mx-auto max-w-xl px-6 pb-20 text-center">
          <h1 className="animate-in fade-in slide-in-from-bottom-3 text-[2rem] leading-[1.1] font-semibold tracking-tight duration-700 sm:text-5xl">
            Everyone deserves
            <br />
            to be{" "}
            <span className="relative whitespace-nowrap">
              heard
              <ThreadUnderline />
            </span>
            .
          </h1>

          <p className="animate-in fade-in slide-in-from-bottom-4 mx-auto mt-5 max-w-md text-sm leading-relaxed text-muted-foreground duration-1000 sm:text-base">
            Speak and follow conversations in real time — read your lips into a
            natural voice, and see who's saying what, live.
          </p>

          <div className="animate-in fade-in slide-in-from-bottom-4 mt-7 flex items-center justify-center gap-2 duration-1000">
            <Link to="/app" className={buttonVariants()}>
              Start speaking
              <ArrowRight />
            </Link>
            <Link
              to="/login"
              className={buttonVariants({ variant: "ghost" })}
            >
              Sign in
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}

function Wordmark() {
  return (
    <Link to="/" className="flex items-center gap-2">
      <span className="flex size-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
        <AudioLines className="size-4" />
      </span>
      <span className="text-base font-semibold tracking-tight">heard</span>
    </Link>
  )
}

/** A little hand-drawn underline using the accent palette. */
function ThreadUnderline() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 200 12"
      preserveAspectRatio="none"
      className="absolute -bottom-1.5 left-0 h-2 w-full"
    >
      <path
        d="M2 7 C 40 2, 70 10, 100 6 S 170 2, 198 7"
        fill="none"
        stroke={ACCENT_COLORS[0]}
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default LandingPage
