import { Component, useState, useEffect, type ReactNode } from 'react';
import './index.css';
import { endpoints } from './api';

import Overview    from './pages/Overview';
import Competitors from './pages/Competitors';
import Features    from './pages/Features';
import SentimentABSA from './pages/SentimentABSA';
import Geospatial  from './pages/Geospatial';
import Networks    from './pages/Networks';
import Scrapers    from './pages/Scrapers';

export const DATASET_CATALOG: Record<string, { label: string; icon: string; desc: string }> = {
  airline:    { label: 'US Airline Sentiment',             icon: '✈️',  desc: '14,640 tweets · 6 airlines · Crowdflower' },
  shopping:   { label: 'Shopping Apps (Amazon/Myntra)',   icon: '🛍️', desc: '99K reviews · e-commerce ratings' },
  indian_air: { label: 'Indian Domestic Airlines',        icon: '🛫', desc: '2.2K reviews · IndiGo + Air India' },
  phones:     { label: 'Consumer Tech (Apple/Samsung)',   icon: '📱', desc: '67.9K reviews · Amazon smartphones' },
  cyber:      { label: 'Twitter Cyberbullying',           icon: '🛡️', desc: '17.8K tweets · multi-class toxicity' },
};

// ─── Nav definition ─────────────────────────────────────────────────────────
const NAV = [
  {
    id: 'overview',
    icon: '📊',
    label: 'Overview',
    badge: 'KPIs',
    chips: ['Net Sentiment','KPI Scores','Pareto 80/20','Trends'],
  },
  {
    id: 'competitors',
    icon: '⚔️',
    label: 'Competitors',
    badge: 'Stats',
    chips: ['NSS Diff','Chi-Square','Voice Share','Aspect Div'],
  },
  {
    id: 'features',
    icon: '📐',
    label: 'Features',
    badge: 'Studio',
    chips: ['Auto Schema','Cramér\'s V','Missingness','Outliers'],
  },
  {
    id: 'sentiment',
    icon: '🎯',
    label: 'Sentiment',
    badge: 'ABSA',
    chips: ['RoBERTa GPU','8-Aspect Matrix','ROC Curves','Live Tester'],
  },
  {
    id: 'geospatial',
    icon: '🗺️',
    label: 'Geospatial',
    badge: 'Maps',
    chips: ['Folium Map','Heatmap','Exact GPS','Regional'],
  },
  {
    id: 'networks',
    icon: '🕸️',
    label: 'Networks',
    badge: 'Graphs',
    chips: ['Echo-Chambers','PageRank','Louvain','Modularity'],
  },
  {
    id: 'scrapers',
    icon: '🕷️',
    label: 'Scrapers',
    badge: 'APIs',
    chips: ['BeautifulSoup','Scrapy','HN Feed','robots.txt'],
  },
];

// ─── Hardware status ─────────────────────────────────────────────────────────
function useHardware() {
  const [hw, setHw] = useState<any>(null);
  useEffect(() => {
    endpoints.hardware()
      .then(r => setHw(r.data))
      .catch(() => setHw({ cuda_available: false, device_name: 'CPU' }));
  }, []);
  return hw;
}

class PageErrorBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidUpdate(prevProps: { resetKey: string }) {
    if (prevProps.resetKey !== this.props.resetKey) this.setState({ error: null });
  }
  render() {
    if (this.state.error) {
      return <div className="alert alert-error">⚠ {this.state.error.message}</div>;
    }
    return this.props.children;
  }
}

const PAGE_MAP: Record<string, React.ComponentType<any>> = {
  overview:    Overview,
  competitors: Competitors,
  features:    Features,
  sentiment:   SentimentABSA,
  geospatial:  Geospatial,
  networks:    Networks,
  scrapers:    Scrapers,
};

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [activeId, setActiveId]     = useState('overview');
  const [activeDsKey, setActiveDsKey] = useState('airline');
  const hw = useHardware();

  const active = NAV.find(n => n.id === activeId) ?? NAV[0];
  const Page = PAGE_MAP[active.id];
  const ds = DATASET_CATALOG[activeDsKey];

  return (
    <div className="app-shell">
      {/* ── Top bar ── */}
      <header className="app-topbar">
        <a className="topbar-logo" href="#" onClick={e => { e.preventDefault(); setActiveId('overview'); }}>
          <span className="topbar-logo-icon">⚡</span>
          BrandPulse AI
        </a>
        <span className="topbar-badge">ENTERPRISE</span>
        <span className="topbar-badge" style={{ background: 'rgba(129,140,248,.1)', color: '#818cf8', borderColor: 'rgba(129,140,248,.3)' }}>
          REACT v2.0
        </span>
        {/* Active dataset pill in topbar */}
        <div style={{ display:'flex', alignItems:'center', gap:6, marginLeft:12 }}>
          <span style={{ fontSize:12, color:'var(--text-dim)', fontWeight:600 }}>Dataset:</span>
          <span style={{ fontSize:12, color:'var(--cyan)', fontWeight:700 }}>{ds.icon} {ds.label}</span>
        </div>

        {/* Hardware pill */}
        <div className="topbar-hw">
          <div className={`hw-dot ${hw?.cuda_available ? '' : 'cpu'}`} />
          {hw
            ? hw.cuda_available
              ? `⚡ ${hw.device_name}  ·  ${hw.vram_total_gb ?? '?'} GB VRAM`
              : '🖥️ CPU Mode'
            : 'Checking hardware…'}
        </div>
      </header>

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-section-label">Navigation</div>
        {NAV.map(item => (
          <div key={item.id}>
            <button
              className={`nav-item ${activeId === item.id ? 'active' : ''}`}
              onClick={() => setActiveId(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
              <span className="nav-badge">{item.badge}</span>
            </button>
            {activeId === item.id && (
              <div className="nav-context-chips">
                {item.chips.map(c => (
                  <span className="context-chip" key={c}>{c}</span>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Dataset info card */}
        <div style={{ marginTop: 'auto', paddingTop: 20 }}>
          <div className="sidebar-ds-card">
            <div className="sidebar-ds-title">{ds.icon} {ds.label}</div>
            <div className="sidebar-ds-meta">{ds.desc}</div>
          </div>
          <div className="sidebar-ds-card" style={{ marginTop: 8 }}>
            <div className="sidebar-ds-title">📡 API Status</div>
            <div className="sidebar-ds-meta">FastAPI · Port 8000 · CORS enabled</div>
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="main-content">
        <PageErrorBoundary resetKey={`${activeId}-${activeDsKey}`}>
          <Page key={`${activeId}-${activeDsKey}`} activeDsKey={activeDsKey} onDsChange={setActiveDsKey} />
        </PageErrorBoundary>
      </main>
    </div>
  );
}
