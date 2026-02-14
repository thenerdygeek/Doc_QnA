import { useEffect, useState } from "react";
import { motion } from "framer-motion";
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

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void;
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
    <div className="relative flex flex-1 flex-col items-center justify-center px-3 sm:px-4">
      {/* Decorative glow orb */}
      <div
        className="pointer-events-none absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2"
        aria-hidden="true"
      >
        <div className="h-40 w-40 rounded-full bg-primary/8 blur-3xl sm:h-64 sm:w-64" />
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
            Get verified answers from your indexed documentation with source
            attribution and confidence scoring.
          </p>
          {stats && (
            <div className="flex items-center justify-center gap-3 pt-1 text-xs text-muted-foreground/70">
              <span className="flex items-center gap-1">
                <Database className="h-3 w-3" />
                {stats.total_files} files indexed
              </span>
              <span
                className="h-3 w-px bg-border"
                aria-hidden="true"
              />
              <span>
                {stats.total_chunks.toLocaleString()} chunks
              </span>
            </div>
          )}
        </motion.div>

        <div className="grid gap-3 sm:grid-cols-2">
          {EXAMPLE_QUESTIONS.map((item, i) => (
            <motion.button
              key={item.label}
              type="button"
              onClick={() => onSelectQuestion(item.question)}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 + i * 0.07, duration: 0.4 }}
              whileHover={{ y: -2, transition: { duration: 0.15 } }}
              className="group flex items-start gap-2.5 rounded-xl border border-border/60 bg-card/50 p-3 text-left shadow-sm transition-colors hover:border-primary/30 hover:bg-card hover:shadow-md sm:gap-3 sm:p-4"
              aria-label={`Ask: ${item.question}`}
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/8 transition-colors group-hover:bg-primary/15">
                <item.icon className="h-4 w-4 text-primary" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-foreground">
                  {item.label}
                </p>
                <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                  {item.question}
                </p>
              </div>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
