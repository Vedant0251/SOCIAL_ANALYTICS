import { useState } from 'react';
import { useApi } from '../useApi';
import { endpoints } from '../api';
import Plot from 'react-plotly.js';

export default function Features() {
  const [ds, setDs] = useState('airline');
  const { data, loading, error } = useApi(() => endpoints.features(ds), [ds]);

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Plus Jakarta Sans', color: '#94a3b8', size: 11 },
    margin: { t: 12, r: 16, b: 50, l: 120 },
    xaxis: { gridcolor: 'rgba(255,255,255,.05)' },
    yaxis: { gridcolor: 'rgba(255,255,255,.05)' },
  };
  const config = { displayModeBar: false, responsive: true };

  const rows = (data as any[]) ?? [];

  // Group by type for donut
  const typeCounts: Record<string, number> = {};
  rows.forEach((r: any) => { const t = r.feature_type ?? r.Type ?? 'Unknown'; typeCounts[t] = (typeCounts[t] ?? 0) + 1; });
  const pieData = [{
    type: 'pie' as const,
    labels: Object.keys(typeCounts),
    values: Object.values(typeCounts),
    hole: 0.55,
    textinfo: 'percent',
    marker: { colors: ['#38bdf8','#818cf8','#34d399','#fb7185','#fbbf24','#e879f9','#2dd4bf'] },
  }];

  // Missing % bar
  const barRows = rows.filter((r: any) => r.missing_pct !== undefined && r.missing_pct !== '');
  const missingBar = [{
    type: 'bar' as const, orientation: 'h' as const,
    y: barRows.map((r: any) => r.column ?? r.Feature ?? ''),
    x: barRows.map((r: any) => parseFloat(r.missing_pct ?? 0)),
    marker: { color: barRows.map((r: any) => parseFloat(r.missing_pct ?? 0) > 30 ? '#fb7185' : '#38bdf8') },
  }];

  const typeColors: Record<string, string> = {
    Nominal: '#38bdf8', Ordinal: '#818cf8', Binary: '#34d399',
    Continuous: '#fbbf24', Discrete: '#fb7185', Datetime: '#e879f9',
    Geospatial: '#2dd4bf', 'Free Text': '#f97316',
  };

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">📐 Feature Intelligence & Studio</h1>
        <p className="page-subtitle">Automated schema classification, type inference, and missingness profiling</p>
        <div className="page-chips">
          {['Auto Schema Classifier','Cramér\'s V','Outlier Detection','Missingness Profile'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      {/* Dataset selector */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="field-label" style={{ marginBottom: 10 }}>Select Dataset</div>
        <div style={{ display: 'flex', gap: 10 }}>
          {[['airline','✈️ US Airline'],['cyber','🛡️ Cyberbullying']].map(([k, label]) => (
            <button key={k} className={`btn ${ds === k ? 'btn-primary' : 'btn-outline'}`} onClick={() => setDs(k)}>{label}</button>
          ))}
        </div>
      </div>

      {loading && <div className="loader"><div className="loader-spinner" /><p className="loader-text">Classifying features…</p></div>}
      {error && <div className="alert alert-error">⚠ {error}</div>}

      {!loading && rows.length > 0 && (
        <>
          {/* Stats */}
          <div className="card-grid card-grid-4" style={{ marginBottom: 20 }}>
            <div className="kpi-card"><div className="kpi-label">Total Features</div><div className="kpi-value" style={{ color: '#38bdf8' }}>{rows.length}</div></div>
            {Object.entries(typeCounts).slice(0, 3).map(([t, n]) => (
              <div key={t} className="kpi-card">
                <div className="kpi-label">{t}</div>
                <div className="kpi-value" style={{ color: typeColors[t] ?? '#94a3b8', fontSize: 28 }}>{n}</div>
                <div className="kpi-caption">columns</div>
              </div>
            ))}
          </div>

          <div className="card-grid card-grid-2" style={{ marginBottom: 20 }}>
            <div className="card">
              <div className="card-header"><div className="card-title">Feature Type Distribution</div></div>
              <Plot data={pieData} layout={{ ...layout, height: 250, margin: { ...layout.margin, l: 20 }, showlegend: true }} config={config} style={{ width: '100%' }} />
            </div>
            <div className="card">
              <div className="card-header"><div className="card-title">Missing Data % per Feature</div></div>
              {barRows.length > 0
                ? <Plot data={missingBar} layout={{ ...layout, height: 250, xaxis: { ...layout.xaxis, ticksuffix: '%', range: [0, 100] } }} config={config} style={{ width: '100%' }} />
                : <div className="alert alert-ok" style={{ margin: 20 }}>✅ No missing data detected</div>
              }
            </div>
          </div>

          {/* Full table */}
          <div className="card">
            <div className="card-header"><div className="card-title">Full Feature Classification Table</div><span className="badge badge-info">{rows.length} columns</span></div>
            <div className="data-table-wrap">
              <table>
                <thead><tr>
                  {Object.keys(rows[0]).filter(k => !['index','Unnamed: 0'].includes(k)).map(k => <th key={k}>{k}</th>)}
                </tr></thead>
                <tbody>{rows.map((row: any, i: number) => (
                  <tr key={i}>{Object.entries(row).filter(([k]) => !['index','Unnamed: 0'].includes(k)).map(([k, v]: any) => (
                    <td key={k}>{k === 'feature_type' || k === 'Type'
                      ? <span className="badge" style={{ background: `${typeColors[v] ?? '#94a3b8'}18`, color: typeColors[v] ?? '#94a3b8', border: `1px solid ${typeColors[v] ?? '#94a3b8'}40` }}>{v}</span>
                      : String(v)
                    }</td>
                  ))}</tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
