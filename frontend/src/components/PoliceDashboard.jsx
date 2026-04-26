import { useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getAllViolations } from '../api'
import axios from 'axios'
import {
  AlertTriangle, Clock, RefreshCw,
  Camera, Upload, Search, Filter, WifiOff,
  Activity, ShieldAlert, ShieldCheck,
} from 'lucide-react'

/* ── Data maps — unchanged from original ── */
const TYPE_LABEL = {
  helmet:    'No Helmet',
  triple:    'Triple Riding',
  overspeed: 'Overspeed',
}

const FINE_MAP = { helmet: 500, triple: 1000, overspeed: 1000 }

const TYPE_BADGE = {
  helmet:    'bg-red-500/10 text-red-400 border-red-500/20',
  triple:    'bg-orange-500/10 text-orange-400 border-orange-500/20',
  overspeed: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
}

const TYPE_DOT = {
  helmet: 'bg-red-400', triple: 'bg-orange-400', overspeed: 'bg-yellow-400',
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

function AnimatedNumber({ value }) {
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    const end = Number(value)
    if (end === 0) { setDisplay(0); return }
    const step  = Math.max(1, Math.ceil(end / 25))
    let current = 0
    const timer = setInterval(() => {
      current += step
      if (current >= end) { setDisplay(end); clearInterval(timer) }
      else setDisplay(current)
    }, 30)
    return () => clearInterval(timer)
  }, [value])
  return <>{display}</>
}

function PageWrapper({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  )
}

/* ── Live Detection Banner ── */
function LiveDetectionBanner({ detection }) {
  const hasViolation = detection.violations_active?.length > 0
  const totalFine    = (detection.violations_active || []).reduce((s, v) => s + (FINE_MAP[v] || 0), 0)

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={hasViolation ? 'alert' : 'idle'}
        initial={{ opacity: 0, scale: 0.99 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className={`relative rounded-2xl border p-4 sm:p-5 mb-5 sm:mb-6
                    overflow-hidden scan-container transition-all duration-500 ${
          hasViolation
            ? 'bg-red-500/[0.07] border-red-500/30 shadow-[0_0_35px_rgba(239,68,68,0.15)]'
            : 'glass border-white/[0.07]'
        }`}
      >
        <div className="scan-line" />

        {/* Header */}
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <div className="flex items-center gap-2">
            <Activity className={`w-4 h-4 ${hasViolation ? 'text-red-400' : 'text-gray-500'}`} />
            <span className="text-xs font-bold tracking-widest text-gray-400 uppercase">
              Live Detection Feed
            </span>
          </div>
          <div className="flex items-center gap-2">
            {hasViolation && (
              <span className="text-xs font-bold text-red-400 bg-red-500/10 border border-red-500/20
                               rounded-full px-2.5 py-0.5 animate-pulse">
                ALERT
              </span>
            )}
            <span className={`w-2.5 h-2.5 rounded-full ${
              hasViolation ? 'bg-red-500 animate-ping' : 'bg-gray-700'
            }`} />
          </div>
        </div>

        {/* Content — stacks on mobile, 3-col on sm+ */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
          <div>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-1.5">
              Detected Plate
            </p>
            <div className={`font-plate text-lg sm:text-xl px-3 sm:px-4 py-1.5 sm:py-2
                             rounded-lg border inline-flex ${
              detection.plate_number
                ? 'text-yellow-400 bg-yellow-400/[0.07] border-yellow-400/30'
                : 'text-gray-700 bg-white/[0.03] border-white/[0.06]'
            }`}>
              {detection.plate_number || '— — — —'}
            </div>
          </div>

          <div className="sm:col-span-2">
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-1.5">
              Active Violations
            </p>
            <div className="flex gap-2 flex-wrap items-center">
              {hasViolation
                ? detection.violations_active.map((v) => (
                    <span key={v} className={`text-xs font-bold px-3 py-1.5 rounded-lg border
                                              ${TYPE_BADGE[v] || 'bg-gray-500/10 text-gray-400 border-gray-500/20'}`}>
                      {TYPE_LABEL[v] || v}
                    </span>
                  ))
                : <span className="text-sm text-gray-700 italic">No violations detected</span>
              }
              {hasViolation && (
                <span className="ml-auto text-orange-400 font-bold text-base">
                  Fine: ₹{totalFine}
                </span>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

export default function PoliceDashboard() {
  /* ── All existing logic unchanged ── */
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
        if (!err.response || err.response.status === 404) setBackendDown(true)
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    let cancelled = false
    const poll = () => {
      axios.get('/detections')
        .then((res) => { if (!cancelled) setDetection(res.data) })
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 1500)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  useEffect(() => {
    fetchViolations()
    const interval = setInterval(fetchViolations, 5000)
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

  const STAT_CARDS = [
    { label: 'Total',   value: stats.total,   color: 'text-white',     Icon: ShieldAlert,   glow: '' },
    { label: 'Pending', value: stats.pending, color: 'text-red-400',   Icon: AlertTriangle, glow: 'hover:shadow-[0_0_20px_rgba(239,68,68,0.12)]' },
    { label: 'Paid',    value: stats.paid,    color: 'text-green-400', Icon: ShieldCheck,   glow: 'hover:shadow-[0_0_20px_rgba(34,197,94,0.12)]' },
  ]

  return (
    <PageWrapper>
      <div className="max-w-7xl mx-auto px-3 sm:px-4 py-5 sm:py-8">

        {/* Backend-down warning */}
        {backendDown && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 sm:mb-5 bg-orange-500/10 border border-orange-500/20 rounded-xl
                       px-4 sm:px-5 py-3 sm:py-4 text-sm flex items-start gap-3"
          >
            <WifiOff className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-orange-400 font-semibold mb-0.5">Backend / Database not reachable</p>
              <p className="text-orange-700">
                Make sure Flask is running on port 9000 and PostgreSQL is up
                (<code className="text-orange-500 font-mono text-xs">docker-compose up -d</code>).
              </p>
            </div>
          </motion.div>
        )}

        {/* Live detection banner */}
        <LiveDetectionBanner detection={detection} />

        {/*
          Layout:
          - Mobile/tablet (< xl): camera full-width, then stats 3-column row, then table
          - Desktop (xl+): camera 2/3 + stats 1/3 side-by-side
        */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 sm:gap-6">

          {/* ── Camera feed ── */}
          <div className="xl:col-span-2">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3
                           flex items-center gap-2">
              <Camera className="w-3.5 h-3.5" /> Live Camera Feed
            </h2>

            <div className="relative rounded-2xl overflow-hidden border border-white/[0.08] bg-[#080810]
                            shadow-[0_0_40px_rgba(240,192,64,0.06)] scan-container">
              <div className="corner-mark tl" />
              <div className="corner-mark tr" />
              <div className="corner-mark bl" />
              <div className="corner-mark br" />

              <img
                src="http://localhost:9000/video"
                alt="Live traffic feed"
                className="w-full block"
                onError={(e) => {
                  e.target.style.display = 'none'
                  e.target.nextSibling.style.display = 'flex'
                }}
              />
              <div className="hidden items-center justify-center h-48 sm:h-64 text-gray-700 flex-col gap-3">
                <Camera className="w-10 h-10 opacity-30" />
                <p className="text-sm font-medium text-gray-600">Camera feed unavailable</p>
                <p className="text-xs text-gray-700">Start the Flask server on port 9000</p>
              </div>
            </div>

            {/* Camera controls — wrap on very small screens */}
            <div className="mt-3 flex flex-wrap gap-2 sm:gap-3">
              <button
                onClick={() => fetch('/use_webcam', { method: 'POST' })}
                className="flex items-center gap-2 px-3.5 sm:px-4 py-2 sm:py-2.5 text-sm
                           glass border-white/[0.08] text-gray-400 rounded-xl
                           hover:border-yellow-400/50 hover:text-yellow-400 transition-all duration-200"
              >
                <Camera className="w-4 h-4 flex-shrink-0" />
                <span>Use Webcam</span>
              </button>
              <label className="flex items-center gap-2 px-3.5 sm:px-4 py-2 sm:py-2.5 text-sm
                                glass border-white/[0.08] text-gray-400 rounded-xl
                                hover:border-yellow-400/50 hover:text-yellow-400
                                transition-all duration-200 cursor-pointer">
                <Upload className="w-4 h-4 flex-shrink-0" />
                <span>Upload Video</span>
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

          {/* ── Stats panel ── */}
          <div className="xl:col-span-1">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3
                           flex items-center gap-2">
              <Activity className="w-3.5 h-3.5" /> Today's Summary
            </h2>

            {/*
              On mobile/tablet: 3 cards in one horizontal row (compact)
              On xl: stacked vertically
            */}
            <div className="grid grid-cols-3 xl:grid-cols-1 gap-3">
              {STAT_CARDS.map(({ label, value, color, Icon, glow }, i) => (
                <motion.div
                  key={label}
                  initial={{ opacity: 0, x: 16 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.07, duration: 0.3 }}
                  className={`glass rounded-xl p-3 sm:p-4 xl:p-5 ${glow} transition-shadow duration-300`}
                >
                  <div className="flex items-center justify-between mb-1.5 sm:mb-2">
                    <p className="text-gray-500 text-[9px] sm:text-xs font-medium uppercase tracking-wider
                                  leading-tight">{label}</p>
                    <Icon className={`w-3.5 h-3.5 ${color} opacity-60 flex-shrink-0`} />
                  </div>
                  {/* Smaller numbers on mobile horizontal layout, larger on xl vertical */}
                  <p className={`text-2xl sm:text-3xl xl:text-4xl font-bold ${color}`}>
                    <AnimatedNumber value={value} />
                  </p>
                </motion.div>
              ))}
            </div>

            <button
              onClick={fetchViolations}
              className="w-full flex items-center justify-center gap-2 mt-3 px-4 py-2.5 sm:py-3 text-sm
                         glass border-white/[0.07] text-gray-400 rounded-xl
                         hover:border-yellow-400/50 hover:text-yellow-400
                         transition-all duration-200 group"
            >
              <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" />
              Refresh Data
            </button>
          </div>
        </div>

        {/* ── Violations section ── */}
        <div className="mt-6 sm:mt-8">

          {/* Filter bar — responsive stacking */}
          <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center
                          justify-between gap-3 sm:gap-4 mb-4">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest
                           flex items-center gap-2 flex-shrink-0">
              <Filter className="w-3.5 h-3.5" /> All Violations
              <span className="text-gray-600 normal-case font-normal tracking-normal">
                ({filtered.length} of {violations.length})
              </span>
            </h2>

            {/* Search + filter buttons wrap together */}
            <div className="flex flex-col xs:flex-row sm:flex-row flex-wrap gap-2 sm:gap-2.5 items-stretch sm:items-center">
              {/* Search — full width on xs, fixed on sm+ */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-600
                                   pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search plate or name…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full sm:w-52 bg-white/[0.04] border border-white/[0.08] rounded-xl
                             pl-9 pr-3 py-2 text-sm text-white placeholder-gray-600
                             focus:outline-none focus:border-yellow-400/50
                             focus:shadow-[0_0_0_3px_rgba(240,192,64,0.08)]
                             transition-all duration-200"
                />
              </div>

              {/* Filter pills */}
              <div className="flex gap-1.5 sm:gap-2">
                {['ALL', 'PENDING', 'PAID'].map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`flex-1 sm:flex-none px-3 sm:px-3.5 py-2 rounded-xl text-xs
                                font-semibold tracking-wide transition-all duration-200 ${
                      filter === f
                        ? 'bg-yellow-400 text-gray-900 shadow-[0_0_18px_rgba(240,192,64,0.3)]'
                        : 'glass border-white/[0.07] text-gray-400 hover:text-yellow-400 hover:border-yellow-400/40'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Table */}
          {loading ? (
            <div className="glass rounded-2xl overflow-hidden">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="px-4 sm:px-5 py-4 border-b border-white/[0.05] flex gap-3 sm:gap-4">
                  {[70, 110, 80, 55, 65, 120].map((w, j) => (
                    <div key={j} className="h-4 bg-white/[0.05] rounded animate-pulse hidden sm:block"
                         style={{ width: w }} />
                  ))}
                  {[80, 90, 60, 55].map((w, j) => (
                    <div key={j} className="h-4 bg-white/[0.05] rounded animate-pulse sm:hidden"
                         style={{ width: w }} />
                  ))}
                </div>
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="glass rounded-2xl p-10 sm:p-12 text-center">
              <Search className="w-8 h-8 text-gray-700 mx-auto mb-3" />
              <p className="text-gray-500 font-medium">No violations match your filter</p>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="glass rounded-2xl overflow-hidden"
            >
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-[480px]">
                  <thead>
                    <tr className="border-b border-white/[0.06] text-gray-500 text-[10px]
                                   uppercase tracking-widest">
                      <th className="px-4 sm:px-5 py-3.5 sm:py-4 text-left font-semibold">Plate</th>
                      {/* Driver col — hidden on small mobile */}
                      <th className="px-4 sm:px-5 py-3.5 sm:py-4 text-left font-semibold hidden sm:table-cell">
                        Driver
                      </th>
                      <th className="px-4 sm:px-5 py-3.5 sm:py-4 text-left font-semibold">Violation</th>
                      <th className="px-4 sm:px-5 py-3.5 sm:py-4 text-left font-semibold">Fine</th>
                      <th className="px-4 sm:px-5 py-3.5 sm:py-4 text-left font-semibold">Status</th>
                      {/* Date col — hidden on small mobile */}
                      <th className="px-4 sm:px-5 py-3.5 sm:py-4 text-left font-semibold hidden md:table-cell">
                        Date & Time
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((v, i) => (
                      <tr
                        key={v.id}
                        className={`border-b border-white/[0.04] table-row-hover transition-colors duration-150 ${
                          i % 2 !== 0 ? 'bg-white/[0.012]' : ''
                        }`}
                      >
                        {/* Plate */}
                        <td className="px-4 sm:px-5 py-3.5 sm:py-4">
                          <span className="font-plate text-yellow-400 text-xs sm:text-sm">
                            {v.plate_number}
                          </span>
                        </td>

                        {/* Driver — hidden on xs */}
                        <td className="px-4 sm:px-5 py-3.5 sm:py-4 text-gray-300 hidden sm:table-cell">
                          {v.user ? (
                            <div>
                              <p className="font-medium text-gray-200 text-sm">{v.user.name}</p>
                              <p className="text-gray-600 text-xs mt-0.5">{v.user.phone}</p>
                            </div>
                          ) : (
                            <span className="text-gray-600 italic text-xs">Unknown</span>
                          )}
                        </td>

                        {/* Violation type */}
                        <td className="px-4 sm:px-5 py-3.5 sm:py-4">
                          <span className={`inline-flex items-center gap-1.5 text-xs font-semibold
                                            px-2 sm:px-2.5 py-1 rounded-md border
                                            ${TYPE_BADGE[v.violation_type] || 'bg-gray-500/10 text-gray-400 border-gray-500/20'}`}>
                            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0
                                              ${TYPE_DOT[v.violation_type] || 'bg-gray-400'}`} />
                            <span className="hidden xs:inline">
                              {TYPE_LABEL[v.violation_type] || v.violation_type}
                            </span>
                            <span className="xs:hidden">
                              {v.violation_type === 'helmet' ? 'Helmet' :
                               v.violation_type === 'triple' ? 'Triple' : 'Speed'}
                            </span>
                          </span>
                        </td>

                        {/* Fine */}
                        <td className="px-4 sm:px-5 py-3.5 sm:py-4">
                          <span className="text-white font-bold text-sm">₹{v.fine_amount}</span>
                        </td>

                        {/* Status */}
                        <td className="px-4 sm:px-5 py-3.5 sm:py-4">
                          <span className={`text-xs font-bold px-2 sm:px-2.5 py-1 rounded-lg border ${
                            v.status === 'PAID'
                              ? 'text-green-400 bg-green-500/10 border-green-500/20'
                              : 'text-red-400 bg-red-500/10 border-red-500/20'
                          }`}>
                            {v.status === 'PAID' ? '✓' : '⏳'}
                            <span className="ml-1 hidden xs:inline">{v.status}</span>
                          </span>
                        </td>

                        {/* Date — hidden on xs/sm */}
                        <td className="px-4 sm:px-5 py-3.5 sm:py-4 text-gray-500 text-xs
                                       hidden md:table-cell">
                          <div className="flex items-center gap-1.5">
                            <Clock className="w-3 h-3 flex-shrink-0" />
                            {formatDate(v.created_at)}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </PageWrapper>
  )
}
