import { useState, useEffect } from 'react';
import axios from 'axios';

export function useApi<T>(fn: () => Promise<{ data: T }>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fn()
      .then(r => { if (!cancelled) setData(r.data); })
      .catch(e => {
        if (!cancelled) {
          if (axios.isAxiosError(e))
            setError(e.response?.data?.detail ?? e.message);
          else setError(String(e));
        }
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error, setData };
}
