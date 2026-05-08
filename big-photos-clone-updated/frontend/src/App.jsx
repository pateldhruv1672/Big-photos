import { useEffect, useMemo, useState } from 'react'
import './styles.css'

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8001'

function absolute(url) {
  if (!url) return ''
  if (url.startsWith('http')) return url
  return `${API}${url}`
}

export default function App() {
  const [items, setItems] = useState([])
  const [stories, setStories] = useState([])
  const [query, setQuery] = useState('beach travel outdoor')
  const [selected, setSelected] = useState(null)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [status, setStatus] = useState('Loading gallery...')
  const [files, setFiles] = useState([])

  async function loadGallery() {
    setStatus('Loading gallery...')
    const res = await fetch(`${API}/api/gallery?limit=40`)
    const data = await res.json()
    setItems(data.items || [])
    setStatus(`Loaded ${data.count || 0} of ${data.total_known || 0} active photos`)
  }

  async function loadStories() {
    try {
      const res = await fetch(`${API}/api/stories`)
      const data = await res.json()
      setStories(data.stories || [])
    } catch (e) {
      console.warn(e)
    }
  }

  async function search() {
    setStatus(`Searching: ${query}`)
    const res = await fetch(`${API}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 60 }),
    })
    const data = await res.json()
    setItems(data.results || [])
    setStatus(`Search returned ${(data.results || []).length} results`)
  }

  async function upload() {
    if (!files.length) return
    const body = new FormData()
    for (const f of files) body.append('files', f)
    setStatus(`Uploading ${files.length} files...`)
    const res = await fetch(`${API}/api/upload`, { method: 'POST', body })
    const data = await res.json()
    setStatus(`Queued ${data.uploaded?.length || 0} uploads. Kafka consumer will label and index them.`)
    setFiles([])
    setTimeout(loadGallery, 1500)
  }

  async function deleteSelected() {
    const imageIds = Array.from(selectedIds)
    if (!imageIds.length) return
    setStatus(`Deleting ${imageIds.length} images...`)
    const res = await fetch(`${API}/api/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds }),
    })
    const data = await res.json()
    setSelectedIds(new Set())
    setItems(items.filter((item) => !imageIds.includes(item.image_id)))
    setStatus(`Deleted ${data.deleted?.length || 0} images and rebuilt active HNSW index`)
  }

  async function refreshBackend() {
    const res = await fetch(`${API}/api/refresh`, { method: 'POST' })
    const data = await res.json()
    setStatus(`Backend refreshed: ${data.metadata_rows} metadata rows, ${data.hnsw_vectors} HNSW vectors`)
    await loadGallery()
    await loadStories()
  }

  useEffect(() => {
    loadGallery()
    loadStories()
  }, [])

  const selectedCount = selectedIds.size

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Big Photos</p>
          <h1>HDFS-backed Google Photos clone</h1>
          <p>Thumbnails and full images are streamed from HDFS through the backend. Search uses hnswlib artifacts stored in HDFS.</p>
        </div>
        <button onClick={refreshBackend}>Refresh backend cache</button>
      </header>

      <section className="toolbar">
        <div className="searchbox">
          <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && search()} />
          <button onClick={search}>Search HNSW</button>
          <button onClick={loadGallery}>Gallery</button>
        </div>
        <div className="uploadbox">
          <input type="file" multiple accept="image/*" onChange={(e) => setFiles(Array.from(e.target.files || []))} />
          <button onClick={upload} disabled={!files.length}>Bulk upload</button>
          <button className="danger" onClick={deleteSelected} disabled={!selectedCount}>Delete selected ({selectedCount})</button>
        </div>
      </section>

      <p className="status">{status}</p>

      <section className="stories">
        {stories.slice(0, 6).map((story) => (
          <article className="story" key={story.story_id}>
            {story.cover_image_url && <img src={absolute(story.cover_image_url)} />}
            <div>
              <h3>{story.title || 'Story'}</h3>
              <p>{story.summary}</p>
            </div>
          </article>
        ))}
      </section>

      <main className="grid">
        {items.map((item) => {
          const checked = selectedIds.has(item.image_id)
          return (
            <article className={`card ${checked ? 'checked' : ''}`} key={item.image_id}>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(e) => {
                    const next = new Set(selectedIds)
                    if (e.target.checked) next.add(item.image_id)
                    else next.delete(item.image_id)
                    setSelectedIds(next)
                  }}
                />
              </label>
              <img src={absolute(item.thumbnail_url)} onClick={() => setSelected(item)} loading="lazy" />
              <div className="meta">
                <h3>{item.category || 'photo'} {item.score ? <span>{item.score.toFixed(2)}</span> : null}</h3>
                <p>{item.caption}</p>
                <div className="chips">{(item.labels || []).slice(0, 4).map((l) => <b key={l}>{l}</b>)}</div>
                <p className="small">Prediction: {item.predicted_label || 'N/A'} {item.predicted_score ? `(${Number(item.predicted_score).toFixed(3)})` : ''}</p>
                <p className="small">Model: {item.model_version || 'N/A'}</p>
                <p className="small">HDFS image: {item.hdfs_image_uri || 'N/A'}</p>
                <p className="small">HDFS thumb: {item.hdfs_thumbnail_uri || 'generated-from-image'}</p>
              </div>
            </article>
          )
        })}
      </main>

      {selected && (
        <div className="modal" onClick={() => setSelected(null)}>
          <div className="modalBody" onClick={(e) => e.stopPropagation()}>
            <button className="close" onClick={() => setSelected(null)}>×</button>
            <img src={absolute(selected.image_url)} />
            <aside>
              <h2>{selected.category}</h2>
              <p>{selected.caption}</p>
              <div className="chips">{(selected.labels || []).map((l) => <b key={l}>{l}</b>)}</div>
              <p className="small">image_id: {selected.image_id}</p>
              <p className="small">Prediction: {selected.predicted_label || 'N/A'} {selected.predicted_score ? `(${Number(selected.predicted_score).toFixed(3)})` : ''}</p>
              <p className="small">Model version: {selected.model_version || 'N/A'}</p>
              <p className="small">HDFS image URI: {selected.hdfs_image_uri || 'N/A'}</p>
              <p className="small">HDFS thumbnail URI: {selected.hdfs_thumbnail_uri || 'generated-from-image'}</p>
              <button className="danger" onClick={async () => { setSelectedIds(new Set([selected.image_id])); await fetch(`${API}/api/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_ids: [selected.image_id] }) }); setSelected(null); await loadGallery(); }}>Delete this photo</button>
            </aside>
          </div>
        </div>
      )}
    </div>
  )
}
