import { useState } from 'react';
import { useApi } from '../useApi';
import { endpoints } from '../api';

export default function Networks() {
  const summary     = useApi(endpoints.netSummary,     []);
  const influencers = useApi(endpoints.netInfluencers,  []);
  const [kind, setKind] = useState<'echo' | 'snap'>('echo');

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">🕸️ Network Dynamics & Echo-Chambers</h1>
        <p className="page-subtitle">PyVis interactive graphs, PageRank influencer leaderboard, Louvain community detection</p>
        <div className="page-chips">
          {['PyVis Echo-Chambers','PageRank Top-N','Louvain Communities','Modularity Q'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      {/* Metrics */}
      {!summary.loading && (summary.data as any[])?.length > 0 && (
        <div className="card-grid card-grid-4" style={{ marginBottom: 20 }}>
          {(summary.data as any[]).slice(0, 4).map((row: any, i: number) => {
            const key = Object.keys(row)[0];
            const val = Object.values(row)[0];
            return (
              <div key={i} className="kpi-card fade-up">
                <div className="kpi-label">{String(key).replace(/_/g,' ')}</div>
                <div className="kpi-value" style={{ fontSize: 24, color: '#818cf8' }}>{String(val)}</div>
              </div>
            );
          })}
        </div>
      )}

      {/* Graph selector + embed */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Interactive Network Graph</div>
            <div className="card-subtitle">Force-directed layout with Louvain community colouring</div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className={`btn ${kind === 'echo' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setKind('echo')}>
              🕸️ Twitter Brand Interaction
            </button>
            <button className={`btn ${kind === 'snap' ? 'btn-primary' : 'btn-outline'}`} onClick={() => setKind('snap')}>
              🌐 Stanford SNAP Ego-Net
            </button>
          </div>
        </div>
        <div className="embed-frame">
          <iframe
            key={kind}
            src={`/api/network/pyvis?kind=${kind}`}
            height={580}
            title="Network Graph"
            style={{ width: '100%', border: 'none', borderRadius: 'var(--radius-md)' }}
          />
        </div>
        <div className="alert alert-info" style={{ marginTop: 12 }}>
          💡 Graph requires <code>pyvis_echo_chamber_network.html</code> — run the pipeline if blank.
        </div>
      </div>

      {/* Influencer Leaderboard */}
      <div className="card">
        <div className="card-header">
          <div><div className="card-title">PageRank Influencer Leaderboard</div><div className="card-subtitle">Top nodes ranked by centrality metrics</div></div>
        </div>
        {influencers.loading && <div className="loader" style={{ minHeight: 80 }}><div className="loader-spinner" /></div>}
        {!influencers.loading && (influencers.data as any[])?.length > 0 && (
          <div className="data-table-wrap">
            <table>
              <thead><tr>
                {Object.keys((influencers.data as any[])[0]).map(k => <th key={k}>{k.replace(/_/g, ' ')}</th>)}
              </tr></thead>
              <tbody>{(influencers.data as any[]).slice(0, 30).map((row: any, i: number) => (
                <tr key={i}>{Object.entries(row).map(([k, v]: any) => (
                  <td key={k}>{typeof v === 'number' ? v.toFixed(4) : String(v)}</td>
                ))}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
        {!influencers.loading && !(influencers.data as any[])?.length && (
          <div className="alert alert-warn">⚠ Influencer table not found. Run the pipeline.</div>
        )}
      </div>
    </div>
  );
}
