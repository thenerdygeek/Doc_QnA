import { motion } from "framer-motion";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorDisplayProps {
  error: string;
  onRetry?: () => void;
}

export function ErrorDisplay({ error, onRetry }: ErrorDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      className="flex items-start gap-3 rounded-xl border border-destructive/20 bg-destructive/5 p-4"
      role="alert"
    >
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-destructive/10">
        <AlertTriangle className="h-4 w-4 text-destructive" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-destructive">
          Something went wrong
        </p>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          {error}
        </p>
      </div>
      {onRetry && (
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          className="shrink-0"
          aria-label="Retry question"
        >
          <RotateCcw className="mr-1.5 h-3 w-3" />
          Retry
        </Button>
      )}
    </motion.div>
  );
}
