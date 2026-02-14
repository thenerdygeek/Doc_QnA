import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { BookOpen, Code, Database, GitCompare, List } from "lucide-react";
import { api } from "@/api/client";
import type { StatsResponse } from "@/types/api";

const EXAMPLE_QUESTIONS = [
  {
    icon: BookOpen,
    label: "Explain a concept",
    question: "How does the authentication flow work?",
  },
  {
    icon: Code,
    label: "Code example",
    question: "Show me an example of the REST API usage",
  },
  {
    icon: GitCompare,
    label: "Compare options",
    question: "What are the differences between v1 and v2 APIs?",
  },
  {
    icon: List,
    label: "Step-by-step",
    question: "How do I set up the development environment?",
  },
];

// ── Typewriter subtitle ────────────────────────────────────────

const SUBTITLE_TEXT = "Ask anything about your documentation";

function TypewriterText({ text }: { text: string }) {
  const prefersReduced = useReducedMotion();
  if (prefersReduced) {
    return <span>{text}</span>;
  }
  return (
    <>
      {text.split("").map((char, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 + i * 0.025, duration: 0.1 }}
        >
          {char}
        </motion.span>
      ))}
    </>
  );
}

// ── Card hover glow (tracks mouse via CSS vars) ─────────────

function useCardGlow() {
  const ref = useRef<HTMLButtonElement>(null);
  function handleMouseMove(e: ReactMouseEvent<HTMLButtonElement>) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    el.style.setProperty("--glow-x", `${e.clientX - rect.left}px`);
    el.style.setProperty("--glow-y", `${e.clientY - rect.top}px`);
  }
  return { ref, handleMouseMove };
}

// ── Container / item variants ──────────────────────────────

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.3 },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 16, scale: 0.96 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { type: "spring" as const, stiffness: 300, damping: 24 },
  },
};

// ── Component ────────────────────────────────────────────────

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void;
}

function GlowCard({
  item,
  onSelect,
}: {
  item: (typeof EXAMPLE_QUESTIONS)[number];
  onSelect: () => void;
}) {
  const { ref, handleMouseMove } = useCardGlow();
  return (
    <motion.button
      ref={ref}
      type="button"
      onClick={onSelect}
      variants={cardVariants}
      whileHover={{ y: -3, transition: { duration: 0.15 } }}
      onMouseMove={handleMouseMove}
      className="card-glow group flex items-start gap-2.5 rounded-xl border border-border/60 bg-card/50 p-3 text-left shadow-sm transition-colors hover:border-primary/30 hover:bg-card hover:shadow-md sm:gap-3 sm:p-4"
      aria-label={`Ask: ${item.question}`}
    >
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/8 transition-colors group-hover:bg-primary/15">
        <item.icon className="h-4 w-4 text-primary" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-foreground">{item.label}</p>
        <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {item.question}
        </p>
      </div>
    </motion.button>
  );
}

export function WelcomeScreen({ onSelectQuestion }: WelcomeScreenProps) {
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    api
      .stats()
      .then((data) => {
        if (mounted) setStats(data);
      })
      .catch(() => {});
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="relative flex flex-1 flex-col items-center justify-center overflow-hidden px-3 sm:px-4">
      {/* Animated gradient orbs */}
      <div className="pointer-events-none absolute inset-0" aria-hidden="true">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <div className="relative max-w-lg space-y-6 text-center sm:space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-3"
        >
          <h2 className="gradient-text text-2xl font-bold tracking-tight sm:text-4xl">
            Ask your docs anything
          </h2>
          <p className="mx-auto max-w-md text-sm leading-relaxed text-muted-foreground">
            <TypewriterText text={SUBTITLE_TEXT} />
          </p>
          {stats && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.4 }}
              className="flex items-center justify-center gap-3 pt-1 text-xs text-muted-foreground/70"
            >
              <span className="flex items-center gap-1">
                <Database className="h-3 w-3" />
                {stats.total_files} files indexed
              </span>
              <span className="h-3 w-px bg-border" aria-hidden="true" />
              <span>{stats.total_chunks.toLocaleString()} chunks</span>
            </motion.div>
          )}
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid gap-3 sm:grid-cols-2"
        >
          {EXAMPLE_QUESTIONS.map((item) => (
            <GlowCard
              key={item.label}
              item={item}
              onSelect={() => onSelectQuestion(item.question)}
            />
          ))}
        </motion.div>
      </div>
    </div>
  );
}
