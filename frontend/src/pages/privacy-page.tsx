import type { ReactNode } from "react"
import { Link, useNavigate } from "react-router-dom"
import { ArrowLeft } from "lucide-react"

import { Button } from "@/components/ui/button"

/** Public privacy policy. Honest about the data heard processes and the third parties involved. */
export function PrivacyPage() {
  const navigate = useNavigate()

  return (
    <div className="no-scrollbar h-svh overflow-y-auto bg-background text-foreground">
      <div className="mx-auto max-w-2xl px-5 py-10">
        <Button
          size="sm"
          variant="outline"
          onClick={() => navigate("/")}
          className="mb-8"
        >
          <ArrowLeft />
          Back
        </Button>

        <h1 className="text-2xl font-semibold tracking-tight">Privacy Policy</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Last updated October 4, 2026
        </p>

        <div className="mt-8 space-y-8 text-sm leading-relaxed">
          <p className="text-muted-foreground">
            heard helps people speak and follow conversations in real time. To do
            that it processes your camera, microphone, and the words exchanged in
            a conversation. This page explains exactly what is collected, how it
            is used, and who it is shared with. heard is an early-stage project —
            please don't share anything you'd consider highly sensitive.
          </p>

          <Section title="What we process">
            <Bullet label="Account">
              When you sign in with Google we receive your name, email address,
              Google account ID, and profile picture. We use these to sign you in
              and label your messages.
            </Bullet>
            <Bullet label="Camera / video">
              Lip reading runs <strong>on your device</strong> in your browser.
              Raw video frames are not uploaded. If you explicitly opt in to
              sharing training clips, short mouth-region crops plus the
              corresponding text may be uploaded to help improve accuracy.
            </Bullet>
            <Bullet label="Microphone / audio">
              In listening mode your microphone audio is streamed to our server
              and on to <strong>ElevenLabs</strong> for speech-to-text and speaker
              diarization. Audio is processed to produce captions; we do not keep
              long-term audio recordings.
            </Bullet>
            <Bullet label="Transcripts & conversations">
              The text produced from your lips and the captions of people around
              you can be saved to your account so you can revisit past
              conversations. These are stored in our database, keyed to your
              account.
            </Bullet>
            <Bullet label="Preferences">
              Your selected voice and similar settings are stored to your account.
            </Bullet>
          </Section>

          <Section title="How we use it">
            <ul className="list-disc space-y-1.5 pl-5 text-muted-foreground">
              <li>Read your lips and speak the result aloud in your chosen voice.</li>
              <li>Transcribe and label the people speaking around you.</li>
              <li>Save and show your past conversations.</li>
              <li>Remember your voice and preferences.</li>
              <li>With your opt-in, improve lip-reading accuracy from shared clips.</li>
            </ul>
          </Section>

          <Section title="Who we share it with">
            <p className="text-muted-foreground">
              We use third-party processors to deliver the service:
            </p>
            <ul className="mt-2 list-disc space-y-1.5 pl-5 text-muted-foreground">
              <li>
                <strong>Google</strong> — sign-in (OAuth).
              </li>
              <li>
                <strong>ElevenLabs</strong> — the text you speak is sent there to
                synthesize a voice, and your microphone audio is sent there for
                speech-to-text. Their handling is governed by their own privacy
                policy.
              </li>
              <li>
                <strong>Our hosting / database provider</strong> — stores your
                saved conversations and preferences.
              </li>
            </ul>
            <p className="mt-2 text-muted-foreground">
              We do not sell your personal data.
            </p>
          </Section>

          <Section title="Retention & your choices">
            <p className="text-muted-foreground">
              Saved conversations and preferences stay until you delete them or
              ask us to remove your account data. Sharing training clips is
              off by default and always opt-in. You can stop using the camera or
              microphone at any time by revoking the browser permission.
            </p>
          </Section>
        </div>

        <div className="mt-10 border-t border-border pt-6 text-xs text-muted-foreground">
          <Link to="/" className="underline underline-offset-4">
            Back to heard
          </Link>
        </div>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h2 className="text-base font-semibold">{title}</h2>
      <div className="mt-2">{children}</div>
    </section>
  )
}

function Bullet({ label, children }: { label: string; children: ReactNode }) {
  return (
    <p className="text-muted-foreground">
      <span className="font-medium text-foreground">{label}.</span> {children}
    </p>
  )
}

export default PrivacyPage
