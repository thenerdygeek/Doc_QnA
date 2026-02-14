import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { ConversationSummary } from "@/types/api";

export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [dbEnabled, setDbEnabled] = useState<boolean | null>(null);
  const hasFetched = useRef(false);

  const fetchList = useCallback(async () => {
    try {
      const list = await api.conversations.list();
      setConversations(list);
      setDbEnabled(true);
    } catch (err) {
      if (err instanceof ApiError && err.status === 501) {
        setDbEnabled(false);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true;
      fetchList();
    }
  }, [fetchList]);

  const refresh = useCallback(() => fetchList(), [fetchList]);

  const deleteConversation = useCallback(
    async (id: string) => {
      try {
        await api.conversations.delete(id);
        setConversations((prev) => prev.filter((c) => c.id !== id));
      } catch (err) {
        console.error("Failed to delete conversation:", err);
      }
    },
    [],
  );

  return {
    conversations,
    loading,
    dbEnabled,
    refresh,
    deleteConversation,
  };
}
