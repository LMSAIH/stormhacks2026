import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom"

import "./index.css"
import { AppPage } from "@/pages/app-page"
import { LabPage } from "@/pages/lab-page"
import { ThemeProvider } from "@/components/theme-provider.tsx"

const router = createBrowserRouter([
  { path: "/", element: <Navigate to="/app" replace /> },
  { path: "/app", element: <AppPage /> },
  { path: "/lab", element: <LabPage /> },
])

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  </StrictMode>
)
