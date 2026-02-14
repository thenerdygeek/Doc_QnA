import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Compass, Menu, MessageSquarePlus, Settings } from "lucide-react";
import { ChatInput } from "@/components/chat-input";
import { MessageList, type Message } from "@/components/message-list";
import { WelcomeScreen } from "@/components/welcome-screen";
import { ThemeToggle } from "@/components/theme-toggle";
import { ConnectionStatus } from "@/components/connection-status";
import { ConversationSidebar } from "@/components/conversation-sidebar";
import { SettingsDialog } from "@/components/settings-dialog";
import { Button } from "@/components/ui/button";
import { useStreamingQuery } from "@/hooks/use-streaming-query";
import { useSession } from "@/hooks/use-session";
import { useConversations } from "@/hooks/use-conversations";
import { useSettings } from "@/hooks/use-settings";
import { useTour } from "@/hooks/use-tour";
import { useIndexing } from "@/hooks/use-indexing";
import { api } from "@/api/client";

const MESSAGES_KEY = "doc-qa-messages";

export default function App() {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = sessionStorage.getItem(MESSAGES_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as Message[];
        return parsed.filter(
          (m) => m.role === "user" || m.content.trim().length > 0,
        );
      }
    } catch {
      // Ignore corrupt data
    }
    return [];
  });
  const stream = useStreamingQuery();
  const { sessionId, setSessionId, clearSession } = useSession();
  const { conversations, dbEnabled, refresh, deleteConversation } =
    useConversations();
  const settings = useSettings();
  const tour = useTour();
  const indexing = useIndexing();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const prevStreamSessionId = useRef<string | null>(null);
  const idPrefix = useId();
  const msgCounter = useRef(messages.length);

  function nextId() {
    return `${idPrefix}-${++msgCounter.current}`;
  }

  // Auto-open settings dialog when tour starts on first visit
  useEffect(() => {
    if (tour.active && tour.isFirstVisit) {
      settings.setOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tour.active]);

  // Persist messages to sessionStorage
  useEffect(() => {
    if (messages.length === 0) {
      sessionStorage.removeItem(MESSAGES_KEY);
    } else {
      sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));
    }
  }, [messages]);

  const submitQuestion = useCallback(
    (question: string) => {
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", content: question },
        { id: nextId(), role: "assistant", content: "" },
      ]);
      stream.submit(question, sessionId);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [stream.submit, sessionId],
  );

  const handleRetry = useCallback(
    (question: string) => {
      setMessages((prev) => prev.slice(0, -1));
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "assistant", content: "" },
      ]);
      stream.submit(question, sessionId);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [stream.submit, sessionId],
  );

  const handleNewChat = useCallback(() => {
    stream.reset();
    setMessages([]);
    clearSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.reset, clearSession]);

  // Sync session ID from stream
  useEffect(() => {
    if (stream.sessionId && stream.sessionId !== prevStreamSessionId.current) {
      prevStreamSessionId.current = stream.sessionId;
      setSessionId(stream.sessionId);
    }
  }, [stream.sessionId, setSessionId]);

  // Persist final answer content back to message for sessionStorage survival
  useEffect(() => {
    if (stream.phase === "complete" && stream.answer) {
      setMessages((prev) =>
        prev.map((msg, i) =>
          msg.role === "assistant" && i === prev.length - 1
            ? { ...msg, content: stream.answer ?? msg.content }
            : msg,
        ),
      );
    }
  }, [stream.phase, stream.answer]);

  // Refresh sidebar after stream completes
  useEffect(() => {
    if (stream.phase === "complete" && dbEnabled) {
      refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.phase, dbEnabled]);

  const handleSelectConversation = useCallback(
    async (id: string) => {
      setSidebarOpen(false);
      try {
        const conv = await api.conversations.get(id);
        const loaded: Message[] = conv.messages.map((m, i) => ({
          id: `loaded-${i}`,
          role: m.role,
          content: m.content,
        }));
        stream.reset();
        setMessages(loaded);
        setSessionId(id);
      } catch (err) {
        console.error("Failed to load conversation:", err);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [stream.reset, setSessionId],
  );

  const handleDeleteConversation = useCallback(
    async (id: string) => {
      await deleteConversation(id);
      // If we deleted the active conversation, start fresh
      if (id === sessionId) {
        handleNewChat();
      }
    },
    [deleteConversation, sessionId, handleNewChat],
  );

  const displayMessages: Message[] = messages.map((msg, i) => {
    if (
      msg.role === "assistant" &&
      i === messages.length - 1 &&
      stream.phase !== "idle"
    ) {
      return { ...msg, streaming: stream };
    }
    return msg;
  });

  const isEmpty = messages.length === 0;
  const showSidebar = dbEnabled === true;

  return (
    <div className="flex h-screen bg-background">
      {/* Conditional sidebar */}
      {showSidebar && (
        <ConversationSidebar
          conversations={conversations}
          activeId={sessionId}
          onSelect={handleSelectConversation}
          onNew={handleNewChat}
          onDelete={handleDeleteConversation}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="glass flex items-center justify-between border-b border-border/60 px-3 py-2 sm:px-6 sm:py-3">
          <div className="flex items-center gap-2">
            {/* Mobile sidebar toggle */}
            {showSidebar && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
                className="text-muted-foreground hover:text-foreground md:hidden"
              >
                <Menu className="h-4 w-4" />
              </Button>
            )}
            <h1 className="gradient-text text-base font-bold tracking-tight sm:text-lg">
              Doc QA
            </h1>
            {!isEmpty && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={handleNewChat}
                aria-label="New conversation"
                className="text-muted-foreground hover:text-foreground"
              >
                <MessageSquarePlus className="h-4 w-4" />
              </Button>
            )}
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => settings.setOpen(true)}
              aria-label="Settings"
              className="text-muted-foreground hover:text-foreground"
            >
              <Settings className="h-4 w-4" />
            </Button>
            <ConnectionStatus />
            <div className="h-4 w-px bg-border" aria-hidden="true" />
            <ThemeToggle />
          </div>
        </header>
        {isEmpty ? (
          <WelcomeScreen onSelectQuestion={submitQuestion} />
        ) : (
          <MessageList messages={displayMessages} onRetry={handleRetry} />
        )}
        <ChatInput
          onSubmit={submitQuestion}
          onStop={stream.cancel}
          isStreaming={stream.phase === "streaming"}
        />
        {/* Bottom "Take a Tour" link for returning users */}
        {!tour.active && (
          <div className="flex justify-center border-t border-border/30 py-1.5">
            <button
              type="button"
              onClick={() => {
                tour.start();
                settings.setOpen(true);
              }}
              className="inline-flex items-center gap-1 text-[11px] text-muted-foreground/60 transition-colors hover:text-muted-foreground"
            >
              <Compass className="h-3 w-3" />
              Take a Tour
            </button>
          </div>
        )}
        <div className="flex justify-end border-t border-border/20 px-4 py-1">
          <span className="text-[10px] text-muted-foreground/40 transition-colors hover:text-muted-foreground/70">
            Made by Subhankar Halder
          </span>
        </div>
      </div>

      <SettingsDialog
        open={settings.open}
        onOpenChange={settings.setOpen}
        settings={settings}
        tour={tour}
        onDbSaved={refresh}
        indexing={indexing}
      />
    </div>
  );
}
