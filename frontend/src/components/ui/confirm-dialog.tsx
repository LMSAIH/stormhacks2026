import { AlertDialog } from "@base-ui/react/alert-dialog"

import { Button } from "@/components/ui/button"

interface ConfirmDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
}

/** A controlled confirmation modal (replaces window.confirm) for destructive actions. */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  onConfirm,
}: ConfirmDialogProps) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm transition-opacity duration-150 data-[starting-style]:opacity-0 data-[ending-style]:opacity-0" />
        <AlertDialog.Popup className="fixed top-1/2 left-1/2 z-50 w-[calc(100vw-2rem)] max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-xl border border-border bg-card p-5 shadow-lg transition-all duration-150 data-[starting-style]:scale-95 data-[starting-style]:opacity-0 data-[ending-style]:scale-95 data-[ending-style]:opacity-0">
          <AlertDialog.Title className="text-sm font-semibold">
            {title}
          </AlertDialog.Title>
          {description && (
            <AlertDialog.Description className="mt-1.5 text-xs text-muted-foreground">
              {description}
            </AlertDialog.Description>
          )}
          <div className="mt-5 flex justify-end gap-2">
            <AlertDialog.Close render={<Button size="sm" variant="outline" />}>
              {cancelLabel}
            </AlertDialog.Close>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => {
                onConfirm()
                onOpenChange(false)
              }}
            >
              {confirmLabel}
            </Button>
          </div>
        </AlertDialog.Popup>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  )
}
