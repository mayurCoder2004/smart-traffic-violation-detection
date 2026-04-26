import { useEffect, useState, useCallback } from 'react'
import { getAllViolations } from '../api'
import axios from 'axios'

const TYPE_LABEL = {
  helmet:    'No Helmet',
  triple:    'Triple Riding',
  overspeed: 'Overspeed',
}

const TYPE_COLOR = {
  helmet:    'text-red-400',
  triple:    'text-orange-400',
  overspeed: 'text-yellow-400',
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

const FINE_MAP  = { helmet: 500, triple: 1000, overspeed: 1000 }
const TYPE_BADGE = {
  helmet:    'bg-red-950 text-red-400 border-red-800',
  triple:    'bg-orange-950 text-orange-400 border-orange-800',
  overspeed: 'bg-yellow-950 text-yellow-400 border-yellow-800',
}

function LiveDetectionBanner({ detection }) {
  const hasViolation = detection.violations_active?.length > 0
  const totalFine    = (detection.violations_active || []).reduce((s, v) => s + (FINE_MAP[v] || 0), 0)

  return (
    <div className={`rounded-xl border p-4 mb-4 transition-all ${
      hasViolation
        ? 'bg-red-950/40 border-red-800'
        : 'bg-[#141414] border-gray-800'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-bold tracking-widest text-gray-500 uppercase">
          Live Detection
        </span>
        <span className={`w-2.5 h-2.5 rounded-full ${
          hasViolation ? 'bg-red-500 animate-pulse' : 'bg-gray-700'
        }`} />
      </div>

      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <span className="text-xs text-gray-600 w-20">Plate</span>
        <span className={`font-mono font-bold tracking-widest text-lg px-3 py-1 rounded-lg border ${
          detection.plate_number
            ? 'text-yellow-400 bg-yellow-950 border-yellow-700'
            : 'text-gray-700 bg-[#111] border-gray-800'
        }`}>
          {detection.plate_number || '—'}
        </span>
      </div>

      <div className="flex items-start gap-3 flex-wrap">
        <span className="text-xs text-gray-600 w-20 mt-1">Violations</span>
        <div className="flex gap-2 flex-wrap">
          {hasViolation
            ? detection.violations_active.map((v) => (
                <span key={v} className={`text-xs font-semibold px-2 py-1 rounded border ${TYPE_BADGE[v] || ''}`}>
                  {TYPE_LABEL[v] || v}
                </span>
              ))
            : <span className="text-sm text-gray-700 italic">No violations detected</span>
          }
        </div>
        {hasViolation && (
          <span className="ml-auto text-orange-400 font-bold text-sm">
            Fine: ₹{totalFine}
          </span>
        )}
      </div>
    </div>
  )
}

export default function PoliceDashboard() {
  const [violations, setViolations]   = useState([])
  const [loading, setLoading]         = useState(true)
  const [backendDown, setBackendDown] = useState(false)
  const [filter, setFilter]           = useState('ALL')
  const [search, setSearch]           = useState('')
  const [detection, setDetection]     = useState({
    violations_active: [], plate_number: null, timestamp: 0,
  })

  const fetchViolations = useCallback(() => {
    getAllViolations()
      .then((res) => { setViolations(res.data); setBackendDown(false) })
      .catch((err) => {
        // 404 means blueprints not registered (DB not running)
        // Network error means Flask not running at all
        if (!err.response || err.response.status === 404) setBackendDown(true)
      })
      .finally(() => setLoading(false))
  }, [])

  // Poll /detections every 1.5 s for live CV output
  // Bug fix: /detections was missing from the Vite proxy — added to vite.config.js
  useEffect(() => {
    let cancelled = false
    const poll = () => {
      axios.get('/detections')
        .then((res) => { if (!cancelled) setDetection(res.data) })
        .catch(() => {})    // silently ignore — banner just stays in idle state
    }
    poll()
    const id = setInterval(poll, 1500)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  useEffect(() => {
    fetchViolations()
    const interval = setInterval(fetchViolations, 5000)   // auto-refresh every 5s
    return () => clearInterval(interval)
  }, [fetchViolations])

  const filtered = violations.filter((v) => {
    const matchFilter = filter === 'ALL' || v.status === filter
    const matchSearch =
      v.plate_number.includes(search.toUpperCase()) ||
      (v.user?.name || '').toLowerCase().includes(search.toLowerCase())
    return matchFilter && matchSearch
  })

  const stats = {
    total:   violations.length,
    pending: violations.filter((v) => v.status === 'PENDING').length,
    paid:    violations.filter((v) => v.status === 'PAID').length,
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">

      {/* Backend-down warning */}
      {backendDown && (
        <div className="mb-4 bg-orange-950 border border-orange-700 rounded-xl px-5 py-4 text-sm">
          <p className="text-orange-400 font-semibold mb-1">Backend / Database not reachable</p>
          <p className="text-orange-700">
            Make sure Flask is running on port 9000 and PostgreSQL is up
            (<code className="text-orange-500">docker-compose up -d</code>).
            Violations table will be empty until the API responds.
          </p>
        </div>
      )}

      {/* Live detection banner — always visible at top */}
      <LiveDetectionBanner detection={detection} />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

        {/* Live camera feed */}
        <div className="xl:col-span-2">
          <h2 className="text-lg font-semibold text-gray-300 mb-3">Live Camera Feed</h2>
          <div className="rounded-2xl overflow-hidden border border-gray-800 bg-[#111] shadow-[0_0_32px_rgba(240,192,64,0.08)]">
            <img
              src="http://localhost:9000/video"
              alt="Live traffic feed"
              className="w-full block"
              onError={(e) => {
                e.target.style.display = 'none'
                e.target.nextSibling.style.display = 'flex'
              }}
            />
            <div
              className="hidden items-center justify-center h-64 text-gray-600 flex-col gap-2"
            >
              <span className="text-4xl">📷</span>
              <p className="text-sm">Camera feed unavailable</p>
              <p className="text-xs text-gray-700">Make sure the Flask server is running on port 9000</p>
            </div>
          </div>

          {/* Camera controls — use relative paths so Vite proxy handles them */}
          <div className="mt-3 flex gap-3">
            <button
              onClick={() => fetch('/use_webcam', { method: 'POST' })}
              className="px-4 py-2 text-sm bg-[#1e1e1e] border border-gray-700 text-gray-300
                         rounded-lg hover:border-yellow-400 hover:text-yellow-400 transition-colors"
            >
              Use Webcam
            </button>
            <label className="px-4 py-2 text-sm bg-[#1e1e1e] border border-gray-700 text-gray-300
                              rounded-lg hover:border-yellow-400 hover:text-yellow-400 transition-colors cursor-pointer">
              Upload Video
              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm"
                className="hidden"
                onChange={async (e) => {
                  const file = e.target.files[0]
                  if (!file) return
                  const fd = new FormData()
                  fd.append('video', file)
                  await fetch('/upload', { method: 'POST', body: fd })
                  e.target.value = ''
                }}
              />
            </label>
          </div>
        </div>

        {/* Stats panel */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-300">Today's Summary</h2>
          {[
            { label: 'Total Violations', value: stats.total,   color: 'text-white' },
            { label: 'Pending Payment',  value: stats.pending, color: 'text-red-400' },
            { label: 'Paid',             value: stats.paid,    color: 'text-green-400' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-[#141414] border border-gray-800 rounded-xl p-5">
              <p className="text-gray-500 text-xs mb-1">{label}</p>
              <p className={`text-4xl font-bold ${color}`}>{value}</p>
            </div>
          ))}

          <button
            onClick={fetchViolations}
            className="w-full mt-2 px-4 py-3 text-sm bg-[#1e1e1e] border border-gray-700
                       text-gray-300 rounded-xl hover:border-yellow-400 hover:text-yellow-400
                       transition-colors"
          >
            Refresh Data
          </button>
        </div>
      </div>

      {/* Violations table */}
      <div className="mt-8">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <h2 className="text-lg font-semibold text-gray-300">All Violations</h2>
          <div className="flex gap-3 flex-wrap">
            <input
              type="text"
              placeholder="Search plate or name..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-[#1e1e1e] border border-gray-700 rounded-lg px-3 py-2 text-sm
                         text-white placeholder-gray-600 focus:outline-none focus:border-yellow-400 w-52"
            />
            {['ALL', 'PENDING', 'PAID'].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === f
                    ? 'bg-yellow-400 text-gray-900'
                    : 'bg-[#1e1e1e] border border-gray-700 text-gray-400 hover:text-yellow-400'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <p className="text-gray-600 text-center py-10">Loading violations...</p>
        ) : filtered.length === 0 ? (
          <div className="bg-[#141414] border border-gray-800 rounded-xl p-10 text-center">
            <p className="text-gray-600">No violations match your filter.</p>
          </div>
        ) : (
          <div className="bg-[#141414] border border-gray-800 rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-wider">
                    <th className="px-5 py-4 text-left">Plate</th>
                    <th className="px-5 py-4 text-left">Driver</th>
                    <th className="px-5 py-4 text-left">Violation</th>
                    <th className="px-5 py-4 text-left">Fine</th>
                    <th className="px-5 py-4 text-left">Status</th>
                    <th className="px-5 py-4 text-left">Date & Time</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((v, i) => (
                    <tr
                      key={v.id}
                      className={`border-b border-gray-800/60 hover:bg-[#1a1a1a] transition-colors ${
                        i % 2 === 0 ? '' : 'bg-[#111]'
                      }`}
                    >
                      <td className="px-5 py-4">
                        <span className="font-mono font-bold text-yellow-400 tracking-wider">
                          {v.plate_number}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-gray-300">
                        {v.user ? (
                          <div>
                            <p className="font-medium">{v.user.name}</p>
                            <p className="text-gray-600 text-xs">{v.user.phone}</p>
                          </div>
                        ) : (
                          <span className="text-gray-600 italic">Unknown</span>
                        )}
                      </td>
                      <td className="px-5 py-4">
                        <span className={`font-semibold ${TYPE_COLOR[v.violation_type] || 'text-gray-400'}`}>
                          {TYPE_LABEL[v.violation_type] || v.violation_type}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-white font-semibold">₹{v.fine_amount}</td>
                      <td className="px-5 py-4">
                        <span className={`text-xs font-bold px-2 py-1 rounded-md ${
                          v.status === 'PAID'
                            ? 'text-green-400 bg-green-950 border border-green-800'
                            : 'text-red-400 bg-red-950 border border-red-800'
                        }`}>
                          {v.status}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-gray-500 text-xs">{formatDate(v.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
