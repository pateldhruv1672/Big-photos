import React, { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import './App.css';

const API = '/api';

// ─── Safe fetch / errors ──────────────────────────────────────────────────────

function detailMessage(detail) {
  if (detail == null) return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail))
    return detail.map((d) => (typeof d === 'object' && d?.msg ? d.msg : JSON.stringify(d))).join('; ');
  if (typeof detail === 'object' && detail.message) return String(detail.message);
  try {
    return JSON.stringify(detail);
  } catch {
    return 'Request failed';
  }
}

async function fetchJsonSafe(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  let parsed = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error(
        res.ok
          ? 'Received invalid JSON from the server.'
          : `Server error (${res.status}). The response was not valid JSON.`,
      );
    }
  }
  if (!res.ok) {
    const msg =
      detailMessage(parsed?.detail) ||
      (typeof parsed?.error === 'string' ? parsed.error : null) ||
      `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return parsed;
}

function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await fetchJsonSafe(url);
      setData(d);
    } catch (e) {
      setError(e.message || 'Something went wrong.');
    }
    setLoading(false);
  }, [url]);
  useEffect(() => {
    refetch();
  }, [refetch]);
  return { data, loading, error, refetch };
}

// ─── Colours ─────────────────────────────────────────────────────────────────

const CAT_COLORS = {
  nature: '#059669',
  travel: '#2563eb',
  city: '#4338ca',
  people: '#be185d',
  food: '#ea580c',
  event: '#ca8a04',
  art: '#7c3aed',
  general: '#475569',
};

// ─── Image thumbnail ─────────────────────────────────────────────────────────

function ImageThumb({ imageId, alt, className }) {
  const [failed, setFailed] = useState(false);
  if (!imageId || failed) {
    return <div className={`thumb-placeholder ${className || ''}`} aria-hidden />;
  }
  return (
    <img
      className={className}
      src={`${API}/image/${encodeURIComponent(imageId)}`}
      alt={alt || ''}
      onError={() => setFailed(true)}
      loading="lazy"
    />
  );
}

function PhotoCard({ photo }) {
  const cat = (photo.category || 'general').toLowerCase();
  const color = CAT_COLORS[cat] || CAT_COLORS.general;
  const conf = photo.confidence ?? photo.quality_score ?? 0;
  const pct = Math.round(Math.min(1, Math.max(0, conf)) * 100);
  const labels = photo.labels || photo.vlm_labels || [];
  const dateStr = photo.taken_at
    ? (() => {
        try {
          const d = new Date(photo.taken_at);
          return Number.isNaN(d.getTime()) ? String(photo.taken_at) : d.toLocaleString();
        } catch {
          return String(photo.taken_at);
        }
      })()
    : null;

  return (
    <article className="photo-card">
      <div className="photo-thumb-wrap">
        <ImageThumb imageId={photo.image_id} alt={photo.caption || photo.file_name} />
        <div className="thumb-badges">
          <span className="badge-cat" style={{ background: color }}>
            {photo.category || 'general'}
          </span>
          <span className="badge-conf">{pct}% conf.</span>
        </div>
      </div>
      <div className="photo-body">
        <p className="photo-caption">{photo.caption || photo.file_name || photo.image_id}</p>
        <div className="pills">
          {labels.slice(0, 6).map((l) => (
            <span key={l} className="pill">
              {l}
            </span>
          ))}
        </div>
        <div className="photo-meta">
          {photo.location ? <span>📍 {photo.location}</span> : null}
          {dateStr ? <span>📅 {dateStr}</span> : null}
          <span className="photo-meta-id">{photo.image_id}</span>
        </div>
      </div>
    </article>
  );
}

// ─── Story card ──────────────────────────────────────────────────────────────

function StoryCard({ story }) {
  const ids = Array.isArray(story.image_ids) ? story.image_ids : [];
  const thumbId = ids[0];
  const coverUri = story.cover_image_uri || '';

  return (
    <article className="story-card">
      <div className="story-thumb">
        {thumbId ? (
          <ImageThumb imageId={thumbId} alt="" />
        ) : (
          <div className="thumb-placeholder" />
        )}
      </div>
      <div className="story-body">
        <h3>{story.title}</h3>
        <p>{story.summary}</p>
        <div className="story-meta">
          {story.location ? <span>📍 {story.location}</span> : null}
          <span>
            🖼 {story.photo_count} photo{story.photo_count !== 1 ? 's' : ''}
          </span>
        </div>
        {coverUri && !thumbId ? (
          <p style={{ fontSize: '0.72rem', color: '#94a3b8', wordBreak: 'break-all', margin: '0 0 0.5rem' }}>
            {coverUri}
          </p>
        ) : null}
        <div className="story-chips">
          {ids.slice(0, 8).map((id) => (
            <span key={id} className="story-chip">
              {id}
            </span>
          ))}
        </div>
      </div>
    </article>
  );
}

// ─── Hero + system chips ─────────────────────────────────────────────────────

function ArchPipeline() {
  const nodes = [
    { cls: 'node-hdfs', icon: '💾', label: 'HDFS' },
    { cls: 'node-kafka', icon: '📡', label: 'Kafka' },
    { cls: 'node-spark', icon: '⚡', label: 'Spark' },
    { cls: 'node-ollama', icon: '🧠', label: 'Ollama VLM' },
    { cls: 'node-ray', icon: '🎯', label: 'Ray Serve' },
    { cls: 'node-hnsw', icon: '🔍', label: 'HNSW' },
    { cls: 'node-react', icon: '🖥', label: 'React UI' },
  ];
  return (
    <div className="arch-pipeline">
      {nodes.map((n, i) => (
        <React.Fragment key={n.label}>
          <div className={`arch-node ${n.cls}`}>
            <span className="node-icon">{n.icon}</span>
            {n.label}
          </div>
          {i < nodes.length - 1 && <span className="arch-arrow">→</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

function HeroSection({ metrics, health }) {
  const basic = metrics?.total_images_basic ?? 0;
  const enriched = metrics?.total_images_enriched ?? 0;
  const hnsw = metrics?.hnsw_index_size ?? 0;
  const kafkaOk = health?.kafka === true;

  const chips = [
    { label: 'HDFS Ready', ok: basic > 0 || enriched > 0 },
    { label: 'Kafka Streaming', ok: kafkaOk },
    { label: 'Spark Analytics', ok: Object.keys(metrics?.category_distribution || {}).length > 0 },
    { label: 'MobileNet V3 Online', ok: enriched > 0 },
    { label: 'HNSW Indexed', ok: hnsw > 0 },
    { label: 'Ollama VLM', ok: true },
  ];

  return (
    <header className="hero">
      <div className="hero-inner">
        <h1>Distributed Photo Intelligence</h1>
        <p className="hero-sub">
          Real-time ML pipeline powered by <strong>HDFS</strong> storage, <strong>Kafka</strong> event streaming,{' '}
          <strong>Spark</strong> analytics, <strong>MobileNet V3</strong> fine-tuned on MIRFLICKR-25K,{' '}
          <strong>Ray Serve</strong> inference, <strong>HNSW</strong> vector search, and <strong>Ollama</strong> VLM enrichment.
        </p>
        <ArchPipeline />
        <div className="system-chips">
          {chips.map((c) => (
            <span key={c.label} className={`chip ${c.ok ? 'ok' : 'warn'}`}>
              <span className="chip-dot" />
              {c.label}
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

function MetricsBar({ metrics, storiesTotal }) {
  if (!metrics) return null;

  const stats = [
    { label: 'Total Photos', value: metrics.total_images_enriched ?? metrics.total_images_basic ?? 0, icon: '🖼' },
    { label: 'HNSW Vectors', value: metrics.hnsw_index_size ?? 0, icon: '🔍' },
    {
      label: 'Categories',
      value: Object.keys(metrics.category_distribution || {}).length,
      icon: '🏷',
    },
    { label: 'Stories', value: storiesTotal ?? 0, icon: '📖' },
  ];

  return (
    <div className="stats-grid">
      {stats.map((s) => (
        <div key={s.label} className="stat-card">
          <div className="stat-icon">{s.icon}</div>
          <div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Gallery ─────────────────────────────────────────────────────────────────

function GalleryTab() {
  const [category, setCategory] = useState('');
  const [offset, setOffset] = useState(0);
  const limit = 12;
  const url = `${API}/gallery?limit=${limit}&offset=${offset}${category ? `&category=${encodeURIComponent(category)}` : ''}`;
  const { data, loading, error } = useApi(url);

  const categories = ['', 'people', 'nature', 'city', 'travel'];

  if (error) {
    return (
      <div className="friendly-error" role="alert">
        {error}
      </div>
    );
  }

  return (
    <div>
      <div className="gallery-filters">
        {categories.map((c) => (
          <button
            key={c || 'all'}
            type="button"
            className={`filter-btn ${category === c ? 'on' : ''}`}
            style={category === c ? { background: CAT_COLORS[c] || '#333', color: '#fff' } : {}}
            onClick={() => {
              setCategory(c);
              setOffset(0);
            }}
          >
            {c || 'All'}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="spinner-wrap">Loading gallery…</div>
      ) : data?.photos?.length === 0 ? (
        <div className="empty-state">
          <div className="big">🖼</div>
          <p>No photos yet. Upload images or run the pipeline.</p>
        </div>
      ) : (
        <>
          <p style={{ color: '#64748b', fontSize: '0.88rem', marginBottom: '1rem' }}>
            Showing {data.photos.length} of {data.total} photos
          </p>
          <div className="photo-grid">{data.photos.map((p) => <PhotoCard key={p.image_id} photo={p} />)}</div>
          <div className="pagination">
            <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
              ← Prev
            </button>
            <span style={{ color: '#64748b', fontSize: '0.88rem' }}>Page {Math.floor(offset / limit) + 1}</span>
            <button type="button" disabled={offset + limit >= data.total} onClick={() => setOffset(offset + limit)}>
              Next →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// ─── Search ───────────────────────────────────────────────────────────────────

function SearchTab() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState('');
  const [err, setErr] = useState(null);

  async function doSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setErr(null);
    try {
      const d = await fetchJsonSafe(`${API}/search?q=${encodeURIComponent(query)}&k=20`);
      setResults(d.results || []);
      setMethod(d.method || '');
    } catch (ex) {
      setErr(ex.message);
      setResults([]);
    }
    setLoading(false);
  }

  return (
    <div>
      <div className="search-hero">
        <h2>Semantic image search</h2>
        <p className="search-sub">Describe scenes, objects, or moods — we match embeddings to your catalog.</p>
      </div>
      <form className="search-row" onSubmit={doSearch}>
        <input
          className="search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: mountain sunset, beach waves, city skyline at night…"
          aria-label="Search query"
        />
        <button type="submit" className="search-btn">
          Search
        </button>
      </form>
      <p className="search-powered">
        Powered by <strong>HNSW vector search</strong>
        {method ? ` · method: ${method}` : ''}
      </p>

      {err ? (
        <div className="friendly-error" style={{ marginTop: '1rem' }} role="alert">
          {err}
        </div>
      ) : null}

      {loading ? <div className="spinner-wrap">Searching…</div> : null}

      {!loading && results !== null && (
        <>
          <p style={{ color: '#64748b', fontSize: '0.88rem', margin: '1rem 0' }}>
            {results.length} result{results.length !== 1 ? 's' : ''}
          </p>
          {results.length === 0 ? (
            <div className="empty-state">
              <div className="big">🔍</div>
              <p>No matches. Try different wording.</p>
            </div>
          ) : (
            <div className="photo-grid">
              {results.map((p, i) => (
                <div key={p.image_id || i} style={{ position: 'relative' }}>
                  {p.distance !== undefined && (
                    <div
                      style={{
                        position: 'absolute',
                        top: 12,
                        right: 12,
                        zIndex: 2,
                        background: 'rgba(15,23,42,0.75)',
                        color: '#fff',
                        borderRadius: 8,
                        padding: '4px 8px',
                        fontSize: 11,
                        fontWeight: 700,
                      }}
                    >
                      Δ {Number(p.distance).toFixed(4)}
                    </div>
                  )}
                  <PhotoCard photo={p} />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {!loading && results === null && !err && (
        <div className="empty-state">
          <div className="big">🔍</div>
          <p>Enter a query above to search your enriched collection.</p>
        </div>
      )}
    </div>
  );
}

// ─── Stories ───────────────────────────────────────────────────────────────────

function StoriesTab() {
  const { data, loading, error } = useApi(`${API}/stories?limit=40`);

  if (loading) return <div className="spinner-wrap">Loading stories…</div>;
  if (error)
    return (
      <div className="friendly-error" role="alert">
        {error}
      </div>
    );

  const stories = data?.stories || [];
  if (stories.length === 0) {
    return (
      <div className="empty-state">
        <div className="big">📖</div>
        <p>No stories yet. Run `make part4` to generate Spark story cards.</p>
      </div>
    );
  }

  return (
    <div className="story-grid">
      {stories.map((s) => (
        <StoryCard key={s.story_id || s.title} story={s} />
      ))}
    </div>
  );
}

// ─── Auto labels ─────────────────────────────────────────────────────────────

function AutoLabelsTab() {
  const [category, setCategory] = useState('');
  const url = `${API}/gallery?limit=40${category ? `&category=${encodeURIComponent(category)}` : ''}`;
  const { data, loading, error } = useApi(url);

  const photos = data?.photos || [];
  const categories = ['people', 'nature', 'city', 'travel'];

  if (loading) return <div className="spinner-wrap">Loading…</div>;
  if (error)
    return (
      <div className="friendly-error" role="alert">
        {error}
      </div>
    );

  return (
    <div>
      <p style={{ color: '#475569', marginBottom: '1.25rem', fontSize: '0.92rem', lineHeight: 1.6 }}>
        Auto-generated labels from VLM enrichment (Ollama when available, deterministic fallback). Each card shows
        scene context, objects, and category.
      </p>
      <div className="gallery-filters" style={{ marginBottom: '1.25rem' }}>
        <button
          type="button"
          className={`filter-btn ${!category ? 'on' : ''}`}
          style={!category ? { background: '#1d4e89', color: '#fff' } : {}}
          onClick={() => setCategory('')}
        >
          All
        </button>
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            className={`filter-btn ${category === c ? 'on' : ''}`}
            style={category === c ? { background: CAT_COLORS[c] || '#1d4e89', color: '#fff' } : {}}
            onClick={() => setCategory(c)}
          >
            {c}
          </button>
        ))}
      </div>

      {photos.length === 0 ? (
        <div className="empty-state">
          <div className="big">🏷</div>
          <p>No labeled photos. Run `make part2` for enrichment.</p>
        </div>
      ) : (
        <div className="photo-grid">
          {photos.map((p) => (
            <PhotoCard key={p.image_id} photo={p} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Upload ──────────────────────────────────────────────────────────────────

function UploadTab() {
  const [status, setStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  async function doUpload(files) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setStatus(null);
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    try {
      const d = await fetchJsonSafe(`${API}/upload`, { method: 'POST', body: fd });
      setStatus({ ok: true, ...d });
    } catch (e) {
      setStatus({ ok: false, message: e.message });
    }
    setUploading(false);
  }

  return (
    <div>
      <label
        className={`upload-zone ${dragOver ? 'drag' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          doUpload(e.dataTransfer.files);
        }}
      >
        <div style={{ fontSize: '2.5rem' }}>📤</div>
        <h3>Drag & drop images here</h3>
        <p>or click to browse · JPG, PNG, WEBP · multiple files</p>
        <input
          type="file"
          multiple
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => doUpload(e.target.files)}
        />
      </label>

      <div className="pipeline-note">
        <strong>Pipeline:</strong> upload → <strong>Kafka</strong> event → consumer writes to{' '}
        <strong>HDFS</strong> → Spark / Ray enrichment → searchable metadata & <strong>HNSW</strong> vectors for this
        dashboard.
      </div>

      {uploading ? (
        <div className="spinner-wrap" style={{ padding: '1.5rem' }}>
          Uploading & queueing…
        </div>
      ) : null}

      {status?.ok ? (
        <div className="status-banner ok">
          <strong>Uploaded {status.uploaded}</strong> image(s). Kafka queued:{' '}
          {status.kafka_queued ? 'yes' : 'no (offline mode)'}.
          {(status.events || []).slice(0, 5).map((ev) => (
            <div key={ev.image_id} style={{ marginTop: '0.35rem', fontSize: '0.82rem' }}>
              • {ev.file_name}
            </div>
          ))}
        </div>
      ) : null}

      {status && !status.ok ? (
        <div className="status-banner err" role="alert">
          {status.message}
        </div>
      ) : null}
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'gallery', label: '🖼  Gallery', Component: GalleryTab },
  { id: 'autolabels', label: '🏷  Auto Labels', Component: AutoLabelsTab },
  { id: 'search', label: '🔍  Search', Component: SearchTab },
  { id: 'stories', label: '📖  Stories', Component: StoriesTab },
  { id: 'upload', label: '📤  Upload', Component: UploadTab },
];

function App() {
  const [tab, setTab] = useState('gallery');
  const Active = TABS.find((t) => t.id === tab)?.Component;
  const { data: metrics } = useApi(`${API}/metrics`);
  const { data: health } = useApi(`${API}/health`);
  const { data: storiesData } = useApi(`${API}/stories?limit=100`);

  const storiesTotal = storiesData?.total ?? (Array.isArray(storiesData?.stories) ? storiesData.stories.length : 0);

  return (
    <div className="app-root">
      <HeroSection metrics={metrics} health={health} />
      <div className="main-wrap">
        <MetricsBar metrics={metrics} storiesTotal={storiesTotal} />
        <div className="tabs-bar">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tab === t.id ? 'active' : ''}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="panel">{Active ? <Active /> : null}</div>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
