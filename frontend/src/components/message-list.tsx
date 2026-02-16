import { useRef, useEffect, useState, memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StreamingAnswer } from "./streaming-answer";
import { SourcesList } from "./sources-list";
import { AttributionList } from "./attribution-list";
import { StatusIndicator } from "./status-indicator";
import { ConfidenceBadge } from "./confidence-badge";
import { ErrorDisplay } from "./error-display";
import type { StreamingQueryState } from "@/hooks/use-streaming-query";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: StreamingQueryState;
}

interface MessageListProps {
  messages: Message[];
  onRetry?: (question: string) => void;
}

// ── Avatar with streaming glow + completion flourish ────────

function AssistantAvatar({ isStreaming, isComplete }: { isStreaming: boolean; isComplete: boolean }) {
  const [flourish, setFlourish] = useState(false);
  const wasStreaming = useRef(false);

  useEffect(() => {
    if (wasStreaming.current && isComplete) {
      setFlourish(true);
      const t = setTimeout(() => setFlourish(false), 500);
      return () => clearTimeout(t);
    }
    wasStreaming.current = isStreaming;
  }, [isStreaming, isComplete]);

  return (
    <motion.div
      animate={flourish ? { scale: [1, 1.15, 1] } : { scale: 1 }}
      transition={flourish ? { duration: 0.4, ease: "easeOut" } : undefined}
      className={[
        "mt-1 hidden h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-primary/5 ring-1 ring-primary/10 sm:flex",
        isStreaming ? "avatar-streaming" : "",
      ].join(" ")}
    >
      <Sparkles className="h-3.5 w-3.5 text-primary" />
    </motion.div>
  );
}

// ── Memoized message item ──────────────────────────────────

interface MessageItemProps {
  msg: Message;
  index: number;
  messages: Message[];
  onRetry?: (question: string) => void;
  prefersReduced: boolean | null;
}

const MessageItem = memo(function MessageItem({ msg, index, messages, onRetry, prefersReduced }: MessageItemProps) {
  const isUser = msg.role === "user";
  const isStreaming = msg.streaming?.phase === "streaming";
  const isComplete = msg.streaming?.phase === "complete";

  return (
    <motion.div
      key={msg.id}
      initial={
        prefersReduced
          ? { opacity: 0 }
          : isUser
            ? { opacity: 0, x: 40 }
            : { opacity: 0, x: -20, scale: 0.97 }
      }
      animate={
        isUser
          ? { opacity: 1, x: 0 }
          : { opacity: 1, x: 0, scale: 1 }
      }
      transition={{ type: "spring", stiffness: 300, damping: 24 }}
      className={isUser ? "flex justify-end" : ""}
      aria-label={isUser ? "Your question" : "Assistant response"}
    >
      {isUser ? (
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-gradient-to-br from-primary to-primary/80 px-3 py-2 text-primary-foreground shadow-md shadow-primary/10 sm:max-w-[80%] sm:px-4 sm:py-2.5">
          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {msg.content}
          </p>
        </div>
      ) : (
        <div className="flex gap-2 sm:gap-3">
          <AssistantAvatar
            isStreaming={!!isStreaming}
            isComplete={!!isComplete}
          />
          <div className="min-w-0 flex-1 space-y-3">
            {msg.streaming && (
              <StatusIndicator
                status={msg.streaming.pipelineStatus}
              />
            )}
            {msg.streaming?.phase === "error" &&
              msg.streaming.error ? (
              <ErrorDisplay
                error={msg.streaming.error}
                onRetry={
                  onRetry && index > 0
                    ? () => onRetry(messages[index - 1]!.content)
                    : undefined
                }
              />
            ) : (
              <StreamingAnswer
                thinkingTokens={msg.streaming?.thinkingTokens ?? ""}
                tokens={msg.streaming?.tokens ?? ""}
                finalAnswer={msg.streaming?.answer ?? msg.content}
                isStreaming={!!isStreaming}
              />
            )}
            {msg.streaming &&
              msg.streaming.sources.length > 0 && (
                <SourcesList sources={msg.streaming.sources} />
              )}
            {msg.streaming &&
              msg.streaming.attributions.length > 0 && (
                <AttributionList
                  attributions={msg.streaming.attributions}
                />
              )}
            <div className="flex flex-wrap items-center gap-3">
              {msg.streaming?.verification && (
                <ConfidenceBadge
                  passed={msg.streaming.verification.passed}
                  confidence={msg.streaming.verification.confidence}
                />
              )}
              {msg.streaming?.phase === "complete" &&
                msg.streaming.elapsed != null && (
                  <span className="text-xs text-muted-foreground">
                    Answered in {msg.streaming.elapsed.toFixed(1)}s
                  </span>
                )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
});

export function MessageList({ messages, onRetry }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const prefersReduced = useReducedMotion();
  const lastMessage = messages[messages.length - 1];
  const latestTokens = lastMessage?.streaming?.tokens;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, latestTokens]);

  return (
    <ScrollArea className="flex-1 px-3 sm:px-4">
      <div
        className="mx-auto max-w-3xl space-y-4 py-4 sm:space-y-6 sm:py-6"
        role="log"
        aria-label="Conversation"
        aria-live="polite"
      >
        {messages.map((msg, i) => (
          <MessageItem
            key={msg.id}
            msg={msg}
            index={i}
            messages={messages}
            onRetry={onRetry}
            prefersReduced={prefersReduced}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
