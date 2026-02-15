import { useCallback, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { MessageSquarePlus, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ConversationSummary } from "@/types/api";

function formatTimeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

interface ConversationSidebarProps {
  conversations: ConversationSummary[];
  activeId?: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  open: boolean;
  onClose: () => void;
}

export function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  open,
  onClose,
}: ConversationSidebarProps) {
  const sidebarRef = useRef<HTMLElement>(null);

  // Focus trap for mobile overlay: keep Tab cycling within sidebar
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab" || !sidebarRef.current) return;

      const focusable = sidebarRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;

      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    },
    [onClose]
  );

  // Auto-focus sidebar when it opens on mobile
  useEffect(() => {
    if (open && sidebarRef.current) {
      const firstButton = sidebarRef.current.querySelector<HTMLElement>("button");
      firstButton?.focus();
    }
  }, [open]);

  const handleDelete = useCallback(
    (e: React.MouseEvent | React.KeyboardEvent, id: string) => {
      e.stopPropagation();
      if (window.confirm("Delete this conversation? This cannot be undone.")) {
        onDelete(id);
      }
    },
    [onDelete]
  );

  return (
    <>
      {/* Mobile overlay backdrop */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/40 md:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar panel */}
      <aside
        ref={sidebarRef}
        onKeyDown={open ? handleKeyDown : undefined}
        className={[
          "flex h-full w-[280px] shrink-0 flex-col border-r border-border/60 bg-card",
          // Mobile: slide-in overlay
          "fixed inset-y-0 left-0 z-50 transition-transform duration-200 md:relative md:z-auto md:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/60 px-3 py-2.5">
          <span className="text-sm font-semibold text-foreground">History</span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onNew}
              aria-label="New conversation"
              className="text-muted-foreground hover:text-foreground"
            >
              <MessageSquarePlus className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onClose}
              aria-label="Close sidebar"
              className="text-muted-foreground hover:text-foreground md:hidden"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Conversation list */}
        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-0.5 p-2">
            <AnimatePresence initial={false}>
              {conversations.map((conv) => (
                <motion.button
                  key={conv.id}
                  layout
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{
                    opacity: 0,
                    x: -20,
                    height: 0,
                    marginTop: 0,
                    transition: { duration: 0.2, ease: "easeIn" },
                  }}
                  transition={{ duration: 0.15 }}
                  onClick={() => onSelect(conv.id)}
                  whileHover={{
                    backgroundColor: conv.id === activeId ? undefined : "var(--accent)",
                    transition: { duration: 0.15 },
                  }}
                  className={[
                    "group relative flex w-full cursor-pointer flex-col gap-0.5 rounded-md px-3 py-2 text-left transition-colors",
                    conv.id === activeId
                      ? "bg-accent text-accent-foreground"
                      : "text-foreground",
                  ].join(" ")}
                >
                  {/* Active indicator bar */}
                  {conv.id === activeId && (
                    <motion.div
                      layoutId="activeSidebarItem"
                      className="absolute left-0 top-1 bottom-1 w-[3px] rounded-full bg-primary"
                      transition={{ type: "spring", stiffness: 400, damping: 28 }}
                    />
                  )}
                  <span className="truncate text-sm font-medium leading-snug">
                    {conv.title || "Untitled"}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatTimeAgo(conv.updated_at)}
                  </span>
                  {/* Delete button — native <button> for accessibility */}
                  <button
                    type="button"
                    aria-label="Delete conversation"
                    onClick={(e) => handleDelete(e, conv.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        handleDelete(e, conv.id);
                      }
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100 focus:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </motion.button>
              ))}
            </AnimatePresence>
            {conversations.length === 0 && (
              <p className="px-3 py-6 text-center text-sm text-muted-foreground">
                No conversations yet
              </p>
            )}
          </div>
        </ScrollArea>
      </aside>
    </>
  );
}
