import { useRef, useEffect } from "react";
import { motion } from "framer-motion";
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

export function MessageList({ messages, onRetry }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
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
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
            className={msg.role === "user" ? "flex justify-end" : ""}
            aria-label={
              msg.role === "user" ? "Your question" : "Assistant response"
            }
          >
            {msg.role === "user" ? (
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-gradient-to-br from-primary to-primary/80 px-3 py-2 text-primary-foreground shadow-md shadow-primary/10 sm:max-w-[80%] sm:px-4 sm:py-2.5">
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                </p>
              </div>
            ) : (
              <div className="flex gap-2 sm:gap-3">
                <div className="mt-1 hidden h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-primary/5 ring-1 ring-primary/10 sm:flex">
                  <Sparkles className="h-3.5 w-3.5 text-primary" />
                </div>
                <div className="min-w-0 flex-1 space-y-3">
                  {msg.streaming && (
                    <StatusIndicator status={msg.streaming.pipelineStatus} />
                  )}
                  {msg.streaming?.phase === "error" && msg.streaming.error ? (
                    <ErrorDisplay
                      error={msg.streaming.error}
                      onRetry={
                        onRetry && i > 0
                          ? () => onRetry(messages[i - 1]!.content)
                          : undefined
                      }
                    />
                  ) : (
                    <StreamingAnswer
                      tokens={msg.streaming?.tokens ?? ""}
                      finalAnswer={msg.streaming?.answer ?? msg.content}
                      isStreaming={msg.streaming?.phase === "streaming"}
                    />
                  )}
                  {msg.streaming && msg.streaming.sources.length > 0 && (
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
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
