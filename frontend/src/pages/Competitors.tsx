import { useState } from 'react';
import { useApi } from '../useApi';
import { endpoints } from '../api';
import Plot from 'react-plotly.js';

export default function Competitors() {
  const { data, loading, error } = useApi(endpoints.competitors, []);
  const sigData = useApi(endpoints.compTable, []);
  const [activePair, setActivePair] = useState<string | null>(null);

  if (loading) return <div className="loader"><div className="loader-spinner" /><p className="loader-text">Running benchmark analysis…</p></div>;
  if (error)   return <div className="alert alert-error">⚠ {error}</div>;

  const pairs = data ? Object.keys(data as Record<string, any>) : [];
  const pair  = activePair ?? pairs[0] ?? null;
  const pd_   = pair && data ? (data as Record<string, any>)[pair] : null;

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Plus Jakarta Sans', color: '#94a3b8', size: 11 },
    margin: { t: 12, r: 16, b: 50, l: 44 },
    xaxis: { gridcolor: 'rgba(255,255,255,.05)' },
    yaxis: { gridcolor: 'rgba(255,255,255,.05)' },
  };
  const config = { displayModeBar: false, responsive: true };

  // NSS comparison bar
  const nssBar = pd_ ? [{
    type: 'bar' as const,
    x: [pd_.brand_a, pd_.brand_b],
    y: [pd_.nss_a ?? 0, pd_.nss_b ?? 0],
    marker: { color: ['#38bdf8', '#818cf8'] },
    text: [`${(pd_.nss_a ?? 0).toFixed(1)}%`, `${(pd_.nss_b ?? 0).toFixed(1)}%`],
    textposition: 'outside',
  }] : [];

  // Voice share pie
  const voicePie = pd_ ? [{
    type: 'pie' as const,
    labels: [pd_.brand_a, pd_.brand_b],
    values: [pd_.volume_a ?? 0, pd_.volume_b ?? 0],
    marker: { colors: ['#38bdf8', '#818cf8'] },
    hole: 0.5, textinfo: 'percent',
  }] : [];

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">⚔️ Competitor Comparison</h1>
        <p className="page-subtitle">Brand-vs-brand NSS, voice share, chi-square significance, and aspect divergence</p>
        <div className="page-chips">
          {['NSS Differential','Chi-Square Test','Voice Share','Aspect Divergence'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      {/* Pair selector */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="field-label">Select Competitor Pair</div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {pairs.map(p => (
            <button
              key={p}
              className={`btn ${p === pair ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setActivePair(p)}
            >{p.replace(/_/g, ' ')}</button>
          ))}
        </div>
      </div>

      {pd_ && (
        <>
          {/* Stats row */}
          <div className="card-grid card-grid-4" style={{ marginBottom: 20 }}>
            {[
              { label: 'NSS Differential', value: `${(pd_.nss_diff ?? 0).toFixed(2)}%`, color: pd_.nss_diff > 0 ? '#34d399' : '#fb7185' },
              { label: 'Chi-Square p-value', value: (pd_.chi2_p ?? 0) < 0.001 ? '< 0.001' : (pd_.chi2_p ?? 0).toFixed(4), color: '#38bdf8' },
              { label: 'Voice Share A', value: `${pd_.brand_a}: ${(pd_.volume_a ?? 0).toLocaleString()}`, color: '#38bdf8' },
              { label: 'Voice Share B', value: `${pd_.brand_b}: ${(pd_.volume_b ?? 0).toLocaleString()}`, color: '#818cf8' },
            ].map(s => (
              <div key={s.label} className="kpi-card fade-up">
                <div className="kpi-label">{s.label}</div>
                <div className="kpi-value" style={{ fontSize: 22, color: s.color }}>{s.value}</div>
              </div>
            ))}
          </div>

          <div className="card-grid card-grid-2" style={{ marginBottom: 20 }}>
            <div className="card">
              <div className="card-header"><div className="card-title">Net Sentiment Score Comparison</div></div>
              <Plot data={nssBar} layout={{ ...layout, height: 250, yaxis: { ...layout.yaxis, ticksuffix: '%' } }} config={config} style={{ width: '100%' }} />
            </div>
            <div className="card">
              <div className="card-header"><div className="card-title">Voice Share</div></div>
              <Plot data={voicePie} layout={{ ...layout, height: 250, showlegend: true }} config={config} style={{ width: '100%' }} />
            </div>
          </div>

          {/* Aspect divergence */}
          {pd_.aspect_comparison && (
            <div className="card">
              <div className="card-header"><div className="card-title">Aspect Divergence Heatmap</div></div>
              <div className="data-table-wrap">
                <table>
                  <thead><tr>
                    <th>Aspect</th>
                    <th>{pd_.brand_a} Score</th>
                    <th>{pd_.brand_b} Score</th>
                    <th>Δ Divergence</th>
                  </tr></thead>
                  <tbody>{(pd_.aspect_comparison as any[]).map((row: any) => (
                    <tr key={row.aspect}>
                      <td>{row.aspect}</td>
                      <td><span style={{ color: row.score_a > 0 ? '#34d399' : '#fb7185' }}>{(row.score_a ?? 0).toFixed(3)}</span></td>
                      <td><span style={{ color: row.score_b > 0 ? '#34d399' : '#fb7185' }}>{(row.score_b ?? 0).toFixed(3)}</span></td>
                      <td><span className={`badge ${Math.abs(row.divergence ?? 0) > 0.2 ? 'badge-neg' : 'badge-info'}`}>{(row.divergence ?? 0).toFixed(3)}</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Significance table */}
      {!sigData.loading && sigData.data && (sigData.data as any[]).length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-header"><div className="card-title">Statistical Significance Tests</div><div className="card-subtitle">Chi-square results across all pairs</div></div>
          <div className="data-table-wrap">
            <table>
              <thead><tr>{Object.keys((sigData.data as any[])[0]).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{(sigData.data as any[]).slice(0,20).map((row: any, i: number) => (
                <tr key={i}>{Object.values(row).map((v: any, j: number) => <td key={j}>{String(v)}</td>)}</tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
