import { useApi } from '../useApi';
import { endpoints } from '../api';

export default function Geospatial() {
  const stats = useApi(endpoints.geoStats, []);

  return (
    <div className="fade-up">
      <div className="page-header">
        <h1 className="page-title">🗺️ Geospatial Intelligence</h1>
        <p className="page-subtitle">Folium MarkerCluster maps, heat density overlays, and regional sentiment breakdown</p>
        <div className="page-chips">
          {['Folium MarkerCluster','Complaint HeatMap','Exact GPS 7.6%','Regional Stats'].map(c =>
            <span className="page-chip" key={c}>{c}</span>)}
        </div>
      </div>

      {/* Interactive Folium Map */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div><div className="card-title">Sentiment MarkerCluster Map</div><div className="card-subtitle">Interactive Folium map — exact GPS + city centroid approximations</div></div>
          <div style={{ display: 'flex', gap: 8 }}>
            <span className="badge badge-ok" style={{ background: 'rgba(52,211,153,.12)', color: '#34d399', border: '1px solid rgba(52,211,153,.2)' }}>● Positive</span>
            <span className="badge badge-neg">● Negative</span>
            <span className="badge badge-neu">● Neutral</span>
          </div>
        </div>
        <div className="embed-frame">
          <iframe
            src="/api/geospatial/folium-map"
            height={520}
            title="Folium Sentiment Map"
            style={{ width: '100%', border: 'none', borderRadius: 'var(--radius-md)' }}
            onError={e => (e.currentTarget.style.display = 'none')}
          />
        </div>
        <div className="alert alert-info" style={{ marginTop: 12 }}>
          💡 Map requires pipeline execution to generate <code>folium_sentiment_map.html</code>. Run <code>python pipeline.py</code> if not visible.
        </div>
      </div>

      {/* Regional Stats Table */}
      <div className="card">
        <div className="card-header">
          <div><div className="card-title">Regional Sentiment Statistics</div><div className="card-subtitle">Aggregated by geographic timezone / state cluster</div></div>
        </div>
        {stats.loading && <div className="loader" style={{ minHeight: 100 }}><div className="loader-spinner" /></div>}
        {stats.error && <div className="alert alert-warn">⚠ {stats.error} — run pipeline to generate regional stats.</div>}
        {!stats.loading && (stats.data as any[])?.length > 0 && (
          <div className="data-table-wrap">
            <table>
              <thead><tr>
                {Object.keys((stats.data as any[])[0]).map(k => <th key={k}>{k.replace(/_/g,' ')}</th>)}
              </tr></thead>
              <tbody>{(stats.data as any[]).map((row: any, i: number) => (
                <tr key={i}>{Object.entries(row).map(([k, v]: any) => (
                  <td key={k}>{k.toLowerCase().includes('pct') || k.toLowerCase().includes('rate')
                    ? `${parseFloat(v).toFixed(1)}%`
                    : String(v)}
                  </td>
                ))}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
        {!stats.loading && !(stats.data as any[])?.length && !stats.error && (
          <div className="alert alert-warn">⚠ Regional stats table not found. Run the pipeline.</div>
        )}
      </div>
    </div>
  );
}
