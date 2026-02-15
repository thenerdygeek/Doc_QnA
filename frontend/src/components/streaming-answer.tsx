import { useRef, useEffect } from "react";
import { Streamdown } from "streamdown";
import { code } from "@streamdown/code";
import { mermaid } from "@streamdown/mermaid";
import "streamdown/styles.css";

const plugins = { code, mermaid };

interface StreamingAnswerProps {
  tokens: string;
  finalAnswer: string | null;
  isStreaming: boolean;
}

export function StreamingAnswer({
  tokens,
  finalAnswer,
  isStreaming,
}: StreamingAnswerProps) {
  const content = isStreaming ? tokens : (finalAnswer ?? "");
  const containerRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLSpanElement>(null);

  // Keep cursor visible by scrolling the container
  useEffect(() => {
    if (!isStreaming || !cursorRef.current) return;
    cursorRef.current.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [isStreaming, content]);

  // Inject copy buttons into code blocks after streaming completes
  useEffect(() => {
    if (isStreaming) return;

    const container = containerRef.current;
    if (!container) return;

    const pres = container.querySelectorAll("pre");
    pres.forEach((pre) => {
      if (pre.querySelector(".copy-btn")) return;

      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.setAttribute("aria-label", "Copy code");
      btn.textContent = "Copy";

      btn.addEventListener("click", async () => {
        const codeEl = pre.querySelector("code");
        const text = codeEl?.textContent ?? pre.textContent ?? "";
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = "Copied!";
          btn.classList.add("copied");
          setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 2000);
        } catch {
          btn.textContent = "Failed";
          setTimeout(() => {
            btn.textContent = "Copy";
          }, 2000);
        }
      });

      pre.appendChild(btn);
    });

    return () => {
      container
        .querySelectorAll(".copy-btn")
        .forEach((btn) => btn.remove());
    };
  }, [isStreaming, content]);

  // Skeleton shimmer while waiting for first token
  if (!content && isStreaming) {
    return (
      <div className="space-y-2.5 py-1" aria-label="Loading response" aria-busy="true" role="status">
        <div className="skeleton-shimmer h-3.5 w-[85%]" />
        <div className="skeleton-shimmer h-3.5 w-[70%]" />
        <div className="skeleton-shimmer h-3.5 w-[60%]" />
      </div>
    );
  }

  if (!content) return null;

  return (
    <div
      ref={containerRef}
      className={`answer-prose ${isStreaming ? "is-streaming" : ""}`}
    >
      <Streamdown plugins={plugins} isAnimating={isStreaming}>
        {content}
      </Streamdown>
      {isStreaming && (
        <span ref={cursorRef} className="streaming-cursor" aria-hidden="true">
          ▍
        </span>
      )}
    </div>
  );
}
