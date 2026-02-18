import { useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ThumbsUp, ThumbsDown, MessageSquare, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/api/client";

interface FeedbackButtonsProps {
  queryId: string;
}

export function FeedbackButtons({ queryId }: FeedbackButtonsProps) {
  const [rating, setRating] = useState<-1 | 0 | 1>(0);
  const [submitting, setSubmitting] = useState(false);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");
  const [commentSent, setCommentSent] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFeedback = async (value: -1 | 1) => {
    if (rating !== 0) return;
    setSubmitting(true);
    try {
      await api.feedback.submit({ query_id: queryId, rating: value });
      setRating(value);
    } catch {
      // silently fail — feedback is non-critical
    } finally {
      setSubmitting(false);
    }
  };

  const handleComment = async () => {
    const trimmed = comment.trim();
    if (!trimmed) return;
    try {
      await api.feedback.submit({ query_id: queryId, rating: rating as -1 | 1, comment: trimmed });
      setCommentSent(true);
      setShowComment(false);
    } catch {
      // silently fail
    }
  };

  return (
    <div className="flex items-center gap-1">
      <AnimatePresence mode="wait">
        {rating === 0 ? (
          <motion.div
            key="buttons"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex gap-0.5"
          >
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => handleFeedback(1)}
              disabled={submitting}
              aria-label="Helpful"
              className="text-muted-foreground hover:text-green-600"
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => handleFeedback(-1)}
              disabled={submitting}
              aria-label="Not helpful"
              className="text-muted-foreground hover:text-red-500"
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </Button>
          </motion.div>
        ) : (
          <motion.div
            key="rated"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-1.5"
          >
            <span className={`text-xs ${rating === 1 ? "text-green-600" : "text-red-500"}`}>
              {rating === 1 ? (
                <ThumbsUp className="inline h-3 w-3" />
              ) : (
                <ThumbsDown className="inline h-3 w-3" />
              )}
            </span>
            {!showComment && !commentSent && (
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={() => {
                  setShowComment(true);
                  setTimeout(() => inputRef.current?.focus(), 50);
                }}
                aria-label="Add comment"
                className="text-muted-foreground hover:text-foreground"
                title="Add a comment"
              >
                <MessageSquare className="h-3 w-3" />
              </Button>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Inline comment input */}
      <AnimatePresence>
        {showComment && (
          <motion.div
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "auto" }}
            exit={{ opacity: 0, width: 0 }}
            className="flex items-center gap-1 overflow-hidden"
          >
            <input
              ref={inputRef}
              type="text"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleComment()}
              placeholder="Optional comment..."
              aria-label="Feedback comment"
              className="h-6 w-40 rounded border border-border/60 bg-background px-2 text-xs text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={handleComment}
              disabled={!comment.trim()}
              aria-label="Send comment"
              className="text-muted-foreground hover:text-primary"
            >
              <Send className="h-3 w-3" />
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
