/**
 * The Agentic Condom: an LLM corrections layer between lip reading and speech (Normal and Quality
 * modes). It may only fix words the reader was unsure of, within a tight budget, and any failure
 * leaves the line as read. Server side: `ml/src/lipread/agentic_condom/`.
 */
export * from "./client"
export * from "./context"
export * from "./gate"
export * from "./settings"
