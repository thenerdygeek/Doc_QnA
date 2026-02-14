import { useCallback, useState } from "react";

const SESSION_KEY = "doc-qa-session-id";

export function useSession() {
  const [sessionId, setSessionIdState] = useState<string | undefined>(() =>
    sessionStorage.getItem(SESSION_KEY) ?? undefined,
  );

  const setSessionId = useCallback((id: string) => {
    sessionStorage.setItem(SESSION_KEY, id);
    setSessionIdState(id);
  }, []);

  const clearSession = useCallback(() => {
    sessionStorage.removeItem(SESSION_KEY);
    setSessionIdState(undefined);
  }, []);

  return { sessionId, setSessionId, clearSession };
}
