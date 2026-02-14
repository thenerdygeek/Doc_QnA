import { motion } from "framer-motion";
import { Quote } from "lucide-react";
import type { AttributionInfo } from "@/types/api";

interface AttributionListProps {
  attributions: AttributionInfo[];
}

export function AttributionList({ attributions }: AttributionListProps) {
  if (attributions.length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
        <Quote className="h-3 w-3" />
        Attributions
      </h4>
      <div className="space-y-1.5">
        {attributions.map((attr, i) => (
          <motion.div
            key={`${attr.source_index}-${i}`}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04, duration: 0.25 }}
            className="flex items-start gap-2.5 rounded-lg border border-border/50 bg-muted/30 px-3 py-2"
          >
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-[10px] font-bold text-primary">
              {attr.source_index + 1}
            </span>
            <p className="flex-1 text-sm leading-relaxed text-foreground/80">
              {attr.sentence}
            </p>
            <span className="shrink-0 font-mono text-xs text-muted-foreground">
              {Math.round(attr.similarity * 100)}%
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
