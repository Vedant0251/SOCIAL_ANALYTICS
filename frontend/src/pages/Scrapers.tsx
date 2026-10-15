import { useState } from 'react';
import { useApi } from '../useApi';
import { endpoints } from '../api';
import api from '../api';

export default function Scrapers() {
  const quotes  = useApi(endpoints.sandboxQuotes, []);
  const hn      = useApi(endpoints.hnFeed, []);
  const [liveData, setLiveData]   = useState<any[] | null>(null);
  const [scraping, setScraping]   = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);

  async function runLive() {
    setScraping(true);
    setLiveError(null);
    try {
      const r = await api.post('/scrapers/live-scrape?pages=2');
      setLiveData(r.data);
    } catch (e: any) {
      setLiveError(e.response?.data?.detail ?? String(e));
    } finally { setScraping(false); }
  }

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">🕷️ Web Scraping & REST APIs</h1>
        <p className="page-subtitle">BeautifulSoup sandbox crawler, Scrapy spider demos, Hacker News live API feed</p>
        <div className="page-chips">
          {['BeautifulSoup','Scrapy Spider','Hacker News API','robots.txt Guard','Live CSV Export'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      {/* Live scrape trigger */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div>
            <div className="card-title">🕷️ Live BeautifulSoup Scraper</div>
            <div className="card-subtitle">Scrapes quotes.toscrape.com with robots.txt compliance and rate limiting</div>
          </div>
          <button className="btn btn-primary" onClick={runLive} disabled={scraping}>
            {scraping ? '⏳ Scraping…' : '⚡ Run Live Scrape (2 pages)'}
          </button>
        </div>
        <div className="alert alert-info">
          ✅ <strong>robots.txt compliant</strong> — only scrapes explicitly allowed paths with 1.5s delay between requests.
        </div>
        {liveError && <div className="alert alert-error">⚠ {liveError}</div>}
        {liveData && liveData.length > 0 && (
          <div className="data-table-wrap" style={{ marginTop: 14 }}>
            <table>
              <thead><tr>{Object.keys(liveData[0]).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{liveData.map((row: any, i: number) => (
                <tr key={i}>{Object.values(row).map((v: any, j: number) => <td key={j}>{String(v)}</td>)}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </div>

      {/* Cached quotes */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div><div className="card-title">📋 Cached Scraped Quotes</div><div className="card-subtitle">Pre-run BeautifulSoup results from sandbox site</div></div>
          {!quotes.loading && <span className="badge badge-info">{(quotes.data as any[])?.length ?? 0} records</span>}
        </div>
        {quotes.loading && <div className="loader" style={{ minHeight: 80 }}><div className="loader-spinner" /></div>}
        {!quotes.loading && (quotes.data as any[])?.length > 0 && (
          <div className="data-table-wrap">
            <table>
              <thead><tr>{Object.keys((quotes.data as any[])[0]).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{(quotes.data as any[]).map((row: any, i: number) => (
                <tr key={i}>{Object.entries(row).map(([k, v]: any) => (
                  <td key={k} style={{ maxWidth: k === 'text' ? 320 : 'none', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: k === 'text' ? 'nowrap' : 'normal' }}>{String(v)}</td>
                ))}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
        {!quotes.loading && !(quotes.data as any[])?.length && (
          <div className="alert alert-warn">⚠ No cached quotes. Run the live scraper above or execute the pipeline.</div>
        )}
      </div>

      {/* Hacker News API */}
      <div className="card">
        <div className="card-header">
          <div><div className="card-title">📡 Hacker News API Feed</div><div className="card-subtitle">Real-time top stories via HN REST API — no authentication required</div></div>
          {!hn.loading && <span className="badge badge-info">{(hn.data as any[])?.length ?? 0} stories</span>}
        </div>
        {hn.loading && <div className="loader" style={{ minHeight: 80 }}><div className="loader-spinner" /></div>}
        {!hn.loading && (hn.data as any[])?.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {(hn.data as any[]).map((story: any, i: number) => (
              <div key={i} style={{
                background: 'var(--bg-raised)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', padding: '12px 16px',
                display: 'flex', alignItems: 'flex-start', gap: 14
              }}>
                <span style={{ color: 'var(--text-dim)', fontSize: 13, fontWeight: 700, minWidth: 24 }}>#{i + 1}</span>
                <div style={{ flex: 1 }}>
                  <a href={story.url ?? `https://news.ycombinator.com/item?id=${story.id}`}
                     target="_blank" rel="noopener noreferrer"
                     style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 14 }}>
                    {story.title}
                  </a>
                  <div style={{ marginTop: 4, display: 'flex', gap: 10, fontSize: 12, color: 'var(--text-dim)' }}>
                    <span>▲ {story.score ?? 0} pts</span>
                    <span>💬 {story.descendants ?? 0} comments</span>
                    <span>by {story.by ?? 'unknown'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
