import { useApi } from '../useApi';
import { endpoints } from '../api';
import Plot from 'react-plotly.js';

function KPI({ label, value, caption, color, icon }: {
  label: string; value: string | number; caption: string; color: string; icon: string;
}) {
  return (
    <div className="kpi-card fade-up">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ color }}>{value}</div>
      <div className="kpi-caption">{caption}</div>
      <div className="kpi-icon">{icon}</div>
      <div className="kpi-accent" style={{ background: color }} />
    </div>
  );
}

function asRows(value: unknown): any[] {
  return Array.isArray(value) ? value : [];
}

export default function Overview({ activeDsKey = 'airline' }: { activeDsKey?: string }) {
  const { data, loading, error } = useApi(() => endpoints.overviewKpis(activeDsKey), [activeDsKey]);

  if (loading) return <div className="loader"><div className="loader-spinner" /><p className="loader-text">Loading KPIs…</p></div>;
  if (error)   return <div className="alert alert-error">⚠ {error}</div>;
  if (!data)   return null;

  const d = data as any;
  const entityCol = d.entity_col ?? 'airline';
  const sentCol = d.sent_col ?? 'airline_sentiment';
  const byEntity = asRows(d.by_entity ?? d.by_airline);
  const pareto = asRows(d.pareto).slice(0, 10);
  const total = Number(d.total_records ?? 0);
  const positive = Number(d.positive ?? 0);
  const negative = Number(d.negative ?? 0);
  const neutral = Number(d.neutral ?? 0);
  const nss = Number(d.nss ?? 0);

  const pieData = [{
    type: 'pie' as const,
    labels: ['Positive', 'Negative', 'Neutral'],
    values: [positive, negative, neutral],
    marker: { colors: ['#34d399', '#fb7185', '#fbbf24'] },
    hole: 0.55,
    textinfo: 'percent',
    hovertemplate: '%{label}: %{value:,} records<extra></extra>',
  }];

  const entities = [...new Set(byEntity.map((r: any) => r[entityCol]).filter(Boolean))];
  const sentiments = ['positive', 'negative', 'neutral'];
  const colors: Record<string, string> = { positive: '#34d399', negative: '#fb7185', neutral: '#fbbf24' };
  const barData = sentiments.map(s => ({
    type: 'bar' as const,
    name: s.charAt(0).toUpperCase() + s.slice(1),
    x: entities,
    y: entities.map(a => {
      const row = byEntity.find((r: any) => r[entityCol] === a && String(r[sentCol]).toLowerCase() === s);
      return row ? row.pct : 0;
    }),
    marker: { color: colors[s] },
  }));

  const paretoBar = [{
    type: 'bar' as const,
    x: pareto.map((r: any) => r.complaint_reason ?? r.negativereason ?? r.category ?? 'Other'),
    y: pareto.map((r: any) => r.count ?? r.volume ?? 0),
    marker: { color: '#38bdf8' },
    name: 'Volume',
  }];
  const paretoCum = [{
    type: 'scatter' as const, mode: 'lines+markers' as const,
    x: pareto.map((r: any) => r.complaint_reason ?? r.negativereason ?? r.category ?? 'Other'),
    y: pareto.map((r: any) => r.cumulative_percentage ?? r.cumulative_pct ?? 0),
    yaxis: 'y2', name: 'Cumulative %',
    line: { color: '#fbbf24', width: 2 },
    marker: { color: '#fbbf24', size: 6 },
  }];

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Plus Jakarta Sans', color: '#94a3b8', size: 11 },
    margin: { t: 12, r: 16, b: 60, l: 44 },
    legend: { orientation: 'h' as const, y: -0.2, font: { size: 11 } },
    xaxis: { gridcolor: 'rgba(255,255,255,.05)', tickangle: -30 },
    yaxis: { gridcolor: 'rgba(255,255,255,.05)' },
  };
  const config = { displayModeBar: false, responsive: true };

  const nssColor = nss > 0 ? '#34d399' : nss < 0 ? '#fb7185' : '#fbbf24';

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">📊 Executive Overview & KPIs</h1>
        <p className="page-subtitle">Real-time health indicators from {total.toLocaleString()} verified social records</p>
        <div className="page-chips">
          {['Net Sentiment Score','Airline Breakdown','Pareto 80/20','Trend Analysis'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      <div className="card-grid card-grid-4" style={{ marginBottom: 24 }}>
        <KPI label="Total Records" value={total.toLocaleString()} caption="Loaded dataset rows" color="#38bdf8" icon="📄" />
        <KPI label="Net Sentiment Score" value={`${nss > 0 ? '+' : ''}${nss}`} caption="(Positive − Negative) / Total × 100" color={nssColor} icon="📊" />
        <KPI label="#1 Complaint Driver" value={d.top_driver ?? 'N/A'} caption={`${d.top_driver_pct ?? 0}% of all negative volume`} color="#fbbf24" icon="⚠️" />
        <KPI label="Negative Rate" value={`${total ? ((negative / total) * 100).toFixed(1) : '0.0'}%`} caption={`${negative.toLocaleString()} negative tweets`} color="#fb7185" icon="📉" />
      </div>

      <div className="card-grid card-grid-2" style={{ marginBottom: 24 }}>
        <div className="card fade-up fade-up-d1">
          <div className="card-header">
            <div><div className="card-title">Sentiment Distribution</div><div className="card-subtitle">3-class polarity across all records</div></div>
          </div>
          <Plot data={pieData} layout={{ ...layout, height: 260, showlegend: true, legend: { orientation: 'h' as const, y: -0.15 } }} config={config} style={{ width: '100%' }} />
        </div>
        <div className="card fade-up fade-up-d2">
          <div className="card-header">
            <div><div className="card-title">Sentiment by Entity</div><div className="card-subtitle">% per class per brand / carrier</div></div>
          </div>
          {entities.length > 0
            ? <Plot data={barData} layout={{ ...layout, height: 260, barmode: 'group' as const, yaxis: { ...layout.yaxis, ticksuffix: '%' } }} config={config} style={{ width: '100%' }} />
            : <div className="alert alert-info">No entity breakdown available for this dataset.</div>}
        </div>
      </div>

      <div className="card fade-up fade-up-d3">
        <div className="card-header">
          <div><div className="card-title">Pareto 80/20 Complaint Drivers</div><div className="card-subtitle">Vital few root causes driving majority of negative volume</div></div>
          <span className="badge badge-info">80/20 Rule</span>
        </div>
        {pareto.length > 0
          ? (
            <Plot
              data={[...paretoBar, ...paretoCum]}
              layout={{
                ...layout, height: 300,
                yaxis2: { overlaying: 'y', side: 'right', ticksuffix: '%', gridcolor: 'transparent', color: '#fbbf24', range: [0, 110] },
                barmode: 'overlay' as const,
              }}
              config={config} style={{ width: '100%' }}
            />
          )
          : <div className="alert alert-info">Pareto drivers are available for the US Airline dataset after the pipeline has run.</div>}
      </div>
    </div>
  );
}
