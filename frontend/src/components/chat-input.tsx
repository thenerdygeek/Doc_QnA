import { useCallback, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSubmit: (question: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

export function ChatInput({
  onSubmit,
  onStop,
  isStreaming,
  disabled,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const [rippling, setRippling] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const triggerRipple = useCallback(() => {
    setRippling(true);
    setTimeout(() => setRippling(false), 600);
  }, []);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    submit();
  }

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    triggerRipple();
    onSubmit(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
    if (e.key === "Escape" && isStreaming) {
      onStop();
    }
  }

  function handleInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <div className="glass border-t border-border/60 pb-[env(safe-area-inset-bottom)]">
      <form
        onSubmit={handleSubmit}
        className="mx-auto flex max-w-3xl items-end gap-2 px-3 py-2 sm:gap-3 sm:px-4 sm:py-3"
        aria-label="Ask a question"
      >
        <div className="input-focus-ring relative flex flex-1 items-end rounded-xl border border-input bg-background shadow-sm transition-all focus-within:border-primary/50">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder="Ask about your docs..."
            disabled={disabled}
            rows={1}
            aria-label="Question input"
            className="flex-1 resize-none bg-transparent px-3 py-2.5 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50 sm:px-4 sm:py-3"
          />
          <div className="p-1.5">
            <AnimatePresence mode="wait" initial={false}>
              {isStreaming ? (
                <motion.div
                  key="stop"
                  initial={{ opacity: 0, rotate: -90, scale: 0.5 }}
                  animate={{ opacity: 1, rotate: 0, scale: 1 }}
                  exit={{ opacity: 0, rotate: 90, scale: 0.5 }}
                  transition={{ duration: 0.2 }}
                >
                  <Button
                    type="button"
                    variant="destructive"
                    size="icon-sm"
                    onClick={onStop}
                    aria-label="Stop generating"
                    className="rounded-lg"
                  >
                    <Square className="h-3.5 w-3.5" />
                  </Button>
                </motion.div>
              ) : (
                <motion.div
                  key="send"
                  initial={{ opacity: 0, rotate: 90, scale: 0.5 }}
                  animate={{ opacity: 1, rotate: 0, scale: 1 }}
                  exit={{ opacity: 0, rotate: -90, scale: 0.5 }}
                  transition={{ duration: 0.2 }}
                >
                  <Button
                    type="submit"
                    size="icon-sm"
                    disabled={!canSend}
                    aria-label="Send question"
                    className={`submit-ripple rounded-lg transition-opacity ${rippling ? "rippling" : ""}`}
                  >
                    <ArrowUp className="h-3.5 w-3.5" />
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </form>
      <p className="mx-auto hidden max-w-3xl px-4 pb-2 text-center text-[11px] text-muted-foreground/60 sm:block">
        Enter to send &middot; Shift+Enter for new line
      </p>
    </div>
  );
}
