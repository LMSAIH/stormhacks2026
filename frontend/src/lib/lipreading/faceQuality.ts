import type { Keypoints } from "./types"

/**
 * Camera settings for lip reading (plan D52): an even ~30 fps so the 25 fps resample has a real
 * frame for every slot (fewer frames cost ~10 WER points at 15 fps), and 640x480 so both face
 * trackers stay cheap while the mouth is still ~60+ px wide at arm's length.
 */
export const CAMERA_CONSTRAINTS: MediaTrackConstraints = {
  facingMode: "user",
  width: { ideal: 640 },
  height: { ideal: 480 },
  frameRate: { ideal: 30, min: 24 },
}

/** Why a frame will read badly; shown to the user so they can fix it. */
export type FaceIssue = "no_face" | "too_small" | "too_dark" | "turned"

export const FACE_ISSUE_TEXT: Record<FaceIssue, string> = {
  no_face: "No face — look at the camera",
  too_small: "Move closer to the camera",
  too_dark: "Too dark — add some light",
  turned: "Face the camera straight on",
}

/** Eye distance below this share of the frame width = face too far away to read lips. */
const MIN_EYE_SHARE = 0.09
/** Mean gray level (0–255) of the frame below this = too dark. */
const MIN_BRIGHTNESS = 60
/** Nose offset from the eyes' midpoint, as a share of eye distance, above this = head turned. */
const MAX_TURN = 0.35

/**
 * Problems with one frame, worst first, from BlazeFace's 4 keypoints (right eye, left eye, nose,
 * mouth) and the frame's mean brightness. Empty = good to read.
 */
export function faceIssues(
  keypoints: Keypoints | null,
  frameWidth: number,
  meanBrightness?: number
): FaceIssue[] {
  const issues: FaceIssue[] = []
  if (meanBrightness !== undefined && meanBrightness < MIN_BRIGHTNESS) issues.push("too_dark")
  if (!keypoints) return ["no_face", ...issues]
  const [rightEye, leftEye, nose] = keypoints
  const eyeDist = Math.hypot(leftEye[0] - rightEye[0], leftEye[1] - rightEye[1])
  if (frameWidth > 0 && eyeDist / frameWidth < MIN_EYE_SHARE) issues.unshift("too_small")
  const midX = (leftEye[0] + rightEye[0]) / 2
  if (eyeDist > 0 && Math.abs(nose[0] - midX) / eyeDist > MAX_TURN) issues.push("turned")
  return issues
}

/** Mean gray level of a frame, sampling every `step`-th pixel (cheap enough per frame). */
export function meanBrightness(gray: Uint8Array, step = 16): number {
  let sum = 0
  let n = 0
  for (let i = 0; i < gray.length; i += step) {
    sum += gray[i]
    n++
  }
  return n ? sum / n : 0
}
