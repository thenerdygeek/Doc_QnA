import { useState } from "react";
import { motion } from "framer-motion";
import { FileText, ExternalLink } from "lucide-react";
import { api } from "@/api/client";
import type { SourceInfo } from "@/types/api";

interface SourcesListProps {
  sources: SourceInfo[];
  highlightedSource?: number | null;
}

function scoreColor(score: number): string {
  if (score >= 0.8) return "bg-emerald-500";
  if (score >= 0.6) return "bg-amber-500";
  return "bg-orange-500";
}

function scoreBarColor(score: number): string {
  if (score >= 0.8) return "bg-emerald-500/60";
  if (score >= 0.6) return "bg-amber-500/60";
  return "bg-orange-500/60";
}

/** Extract just the filename from a full path. */
function fileName(filePath: string): string {
  return filePath.split("/").pop() ?? filePath;
}

export function SourcesList({ sources, highlightedSource }: SourcesListProps) {
  const [opening, setOpening] = useState<number | null>(null);

  if (sources.length === 0) return null;

  async function handleOpen(filePath: string, index: number) {
    setOpening(index);
    try {
      const result = await api.files.open(filePath);
      if (!result.ok) {
        console.warn("Failed to open file:", result.error);
      }
    } catch (err) {
      console.warn("Failed to open file:", err);
    } finally {
      setTimeout(() => setOpening(null), 600);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="space-y-2.5"
    >
      <h4 className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
        <FileText className="h-3 w-3" />
        Sources
        <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-bold text-primary">
          {sources.length}
        </span>
      </h4>
      <div className="grid gap-2 sm:grid-cols-2">
        {sources.map((source, i) => (
          <motion.button
            key={i}
            id={`source-card-${i + 1}`}
            type="button"
            onClick={() => handleOpen(source.file_path, i)}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04, duration: 0.25 }}
            whileHover={{ y: -1, transition: { duration: 0.15 } }}
            whileTap={{ scale: 0.98 }}
            className={`group flex w-full cursor-pointer items-start gap-2 rounded-lg border bg-card/50 p-2.5 text-left transition-all hover:border-primary/40 hover:bg-primary/5 hover:shadow-md sm:gap-2.5 sm:p-3 ${
              highlightedSource === i + 1
                ? "border-primary/60 ring-2 ring-primary/20 bg-primary/5"
                : "border-border/60"
            }`}
          >
            <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-primary/8 transition-colors group-hover:bg-primary/15">
              <FileText className="h-3 w-3 text-primary" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-foreground">
                {source.section_title}
              </p>
              <p className="truncate font-mono text-xs text-muted-foreground">
                {fileName(source.file_path)}
              </p>
            </div>
            <div className="flex shrink-0 flex-col items-end gap-1">
              <div className="flex items-center gap-1.5">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${scoreColor(source.score)}`}
                />
                <span className="font-mono text-xs text-muted-foreground">
                  {Math.round(source.score * 100)}%
                </span>
                <ExternalLink
                  className={`ml-0.5 h-3 w-3 transition-all ${
                    opening === i
                      ? "text-primary"
                      : "text-transparent group-hover:text-muted-foreground"
                  }`}
                />
              </div>
              {/* Score bar */}
              <div className="h-1 w-12 overflow-hidden rounded-full bg-muted/50">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.round(source.score * 100)}%` }}
                  transition={{ delay: 0.2 + i * 0.05, duration: 0.5, ease: "easeOut" }}
                  className={`h-full rounded-full ${scoreBarColor(source.score)}`}
                />
              </div>
            </div>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
