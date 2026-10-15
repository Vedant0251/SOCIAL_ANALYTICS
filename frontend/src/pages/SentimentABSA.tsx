import { useState } from 'react';
import { useApi } from '../useApi';
import { endpoints } from '../api';
import Plot from 'react-plotly.js';

const ASPECTS = ['Seat Comfort','Cabin Service','Ground Service','Food & Beverage','Entertainment','Booking','Baggage','Value'];

function AbsaAspectCard({ name, score }: { name: string; score: number | null }) {
  const color = score == null ? '#475569' : score > 0.1 ? '#34d399' : score < -0.1 ? '#fb7185' : '#fbbf24';
  return (
    <div className="absa-aspect-card">
      <div className="absa-aspect-name">{name}</div>
      <div className="absa-aspect-value" style={{ color }}>{score == null ? 'N/A' : score.toFixed(3)}</div>
      <div className="absa-detected" style={{ color }}>
        {score == null ? '—' : score > 0.1 ? '▲ Positive' : score < -0.1 ? '▼ Negative' : '● Neutral'}
      </div>
    </div>
  );
}

function LiveTester() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [absa, setAbsa] = useState<any>(null);
  const [running, setRunning] = useState(false);

  async function run() {
    if (!text.trim()) return;
    setRunning(true);
    try {
      const [r1, r2] = await Promise.all([endpoints.predict(text), endpoints.absaPredict(text)]);
      setResult(r1.data);
      setAbsa(r2.data);
    } catch (e: any) {
      console.error(e);
    } finally { setRunning(false); }
  }

  const labelColor: Record<string, string> = { positive: '#34d399', negative: '#fb7185', neutral: '#fbbf24' };
  const fillClass: Record<string, string>  = { positive: 'fill-pos', negative: 'fill-neg', neutral: 'fill-neu' };

  return (
    <div className="card">
      <div className="card-header">
        <div><div className="card-title">✍️ Live Text Tester</div><div className="card-subtitle">GPU-accelerated RoBERTa + ABSA inference</div></div>
        {running && <div style={{ width: 20, height: 20, border: '2px solid rgba(56,189,248,.2)', borderTopColor: '#38bdf8', borderRadius: '50%', animation: 'spin .8s linear infinite' }} />}
      </div>
      <div className="field-label">Enter text to analyse</div>
      <textarea rows={4} value={text} onChange={e => setText(e.target.value)}
        placeholder="e.g. The legroom was terrible but the crew was incredibly friendly and efficient." />
      <button className="btn btn-primary" style={{ marginTop: 12 }} onClick={run} disabled={running || !text.trim()}>
        {running ? '⏳ Analysing…' : '⚡ Run Analysis'}
      </button>

      {result && (
        <div className="predict-result">
          <div className="predict-label-row">
            <div className="predict-label" style={{ color: labelColor[result.label] ?? '#fff' }}>
              {result.label?.toUpperCase()}
            </div>
            <span className="badge badge-info">{(result.confidence * 100).toFixed(1)}% confidence</span>
          </div>
          {['positive','negative','neutral'].map(cls => (
            <div key={cls} className="confidence-bar-wrap">
              <span style={{ width: 60, color: labelColor[cls] }}>{cls}</span>
              <div className="confidence-bar">
                <div className={`confidence-fill ${fillClass[cls]}`}
                  style={{ width: `${((result.scores?.[cls] ?? 0) * 100).toFixed(1)}%` }} />
              </div>
              <span>{((result.scores?.[cls] ?? 0) * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {absa && (
        <div style={{ marginTop: 16 }}>
          <div className="field-label" style={{ marginBottom: 8 }}>Aspect-Based Scores</div>
          <div className="absa-grid">
            {Object.entries(absa).map(([asp, sc]: any) => (
              <AbsaAspectCard key={asp} name={asp} score={sc} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function SentimentABSA() {
  const matrixData = useApi(endpoints.absaMatrix, []);
  const metricsData = useApi(endpoints.sentMetrics, []);

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Plus Jakarta Sans', color: '#94a3b8', size: 11 },
    margin: { t: 12, r: 16, b: 60, l: 100 },
    xaxis: { gridcolor: 'rgba(255,255,255,.05)' },
    yaxis: { gridcolor: 'rgba(255,255,255,.05)' },
  };
  const config = { displayModeBar: false, responsive: true };

  // Build heatmap from ABSA matrix
  const matrix = (matrixData.data as any[]) ?? [];
  const heatAirlines = [...new Set(matrix.map((r: any) => r.airline))];
  const heatAspects  = ASPECTS;
  const zMatrix = heatAspects.map(asp =>
    heatAirlines.map(air => {
      const row = matrix.find((r: any) => r.airline === air && r.aspect === asp);
      return row ? parseFloat(row.avg_score ?? row.score ?? 0) : 0;
    })
  );
  const heatmapData = [{
    type: 'heatmap' as const,
    x: heatAirlines, y: heatAspects, z: zMatrix,
    colorscale: [[0,'#fb7185'],[0.5,'#1e293b'],[1,'#34d399']] as any,
    zmid: 0, zmin: -1, zmax: 1,
    hoverongaps: false,
    hovertemplate: '<b>%{x}</b> — %{y}<br>Score: %{z:.3f}<extra></extra>',
  }];

  const metrics = (metricsData.data as any[]) ?? [];

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">🎯 Aspect Sentiment & Deep ML</h1>
        <p className="page-subtitle">RoBERTa GPU inference, 8-aspect ABSA matrix, model performance curves, and live tester</p>
        <div className="page-chips">
          {['RoBERTa GPU','8-Aspect Matrix','ROC Curves','Live Tester'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      {/* ABSA Heatmap */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div><div className="card-title">8-Aspect Polarity Matrix</div><div className="card-subtitle">Per-airline aspect net polarity score (green = positive, red = negative)</div></div>
          <span className="badge badge-info">ABSA</span>
        </div>
        {matrixData.loading && <div className="loader" style={{ minHeight: 120 }}><div className="loader-spinner" /></div>}
        {!matrixData.loading && matrix.length > 0 && (
          <Plot data={heatmapData} layout={{ ...layout, height: 320, margin: { ...layout.margin, l: 130 } }} config={config} style={{ width: '100%' }} />
        )}
        {!matrixData.loading && matrix.length === 0 && (
          <div className="alert alert-warn">⚠ ABSA matrix not found. Run the pipeline to generate it.</div>
        )}
      </div>

      {/* Model metrics */}
      {metrics.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header"><div className="card-title">Model Summary Metrics</div></div>
          <div className="card-grid card-grid-4">
            {metrics.map((m: any, i: number) => (
              <div key={i} className="kpi-card">
                <div className="kpi-label">{m.metric ?? m.Metric ?? `Metric ${i}`}</div>
                <div className="kpi-value" style={{ fontSize: 24, color: '#38bdf8' }}>{m.value ?? m.Value ?? '—'}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live tester */}
      <LiveTester />
    </div>
  );
}
