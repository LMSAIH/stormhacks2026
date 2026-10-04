import { Moon, Sun } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"

/** Floating light/dark switch. Icon swaps automatically with the `.dark` class. */
export function ThemeToggle() {
  const { setTheme } = useTheme()

  const toggle = () => {
    const isDark = document.documentElement.classList.contains("dark")
    setTheme(isDark ? "light" : "dark")
  }

  return (
    <Button
      size="icon"
      variant="outline"
      onClick={toggle}
      aria-label="Toggle light and dark mode"
      className="bg-card/80 backdrop-blur"
    >
      <Moon className="dark:hidden" />
      <Sun className="hidden dark:block" />
    </Button>
  )
}
