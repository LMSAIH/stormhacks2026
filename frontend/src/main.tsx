import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom"

import "./index.css"
import { AppPage } from "@/pages/app-page"
import { NotesPage } from "@/pages/notes-page"
import { NoteDetailPage } from "@/pages/note-detail-page"
import { LabPage } from "@/pages/lab-page"
import { LoginPage } from "@/pages/login-page"
import { RequireAuth } from "@/components/app/require-auth"
import { AuthProvider } from "@/lib/backend/auth-context"
import { ThemeProvider } from "@/components/theme-provider.tsx"

const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  {
    // Everything below requires sign-in; signed-out users land on /login.
    element: <RequireAuth />,
    children: [
      { path: "/", element: <Navigate to="/app" replace /> },
      { path: "/app", element: <AppPage /> },
      { path: "/notes", element: <NotesPage /> },
      { path: "/notes/:id", element: <NoteDetailPage /> },
      { path: "/lab", element: <LabPage /> },
    ],
  },
])

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <RouterProvider router={router} />
      </AuthProvider>
    </ThemeProvider>
  </StrictMode>
)
