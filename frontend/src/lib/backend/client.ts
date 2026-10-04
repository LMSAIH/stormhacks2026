import { API_URL } from "./config"

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

/** GET a JSON endpoint with the session cookie attached. */
export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { credentials: "include" })
  if (!res.ok) throw new ApiError(res.status, `GET ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

/** PUT JSON with the session cookie attached. */
export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  return apiSend<T>("PUT", path, body)
}

/** POST JSON with the session cookie attached. */
export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return apiSend<T>("POST", path, body)
}

/** DELETE with the session cookie attached. */
export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!res.ok) throw new ApiError(res.status, `DELETE ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

async function apiSend<T>(
  method: "POST" | "PUT",
  path: string,
  body: unknown
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new ApiError(res.status, `${method} ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

export function isUnauthorized(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401
}
