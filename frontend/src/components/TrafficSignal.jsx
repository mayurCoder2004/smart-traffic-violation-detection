import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import {
  TrafficCone, Upload, Camera, RefreshCw,
  Circle, AlertCircle, CheckCircle2, Timer,
  Car, Truck, Bike,
} from 'lucide-react'

const API = 'http://localhost:9000'
const POLL_MS = 800

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

/* ── Countdown ring SVG ── */
function CountdownRing({ remaining, total, color }) {
  const R    = 28
  const CIRC = 2 * Math.PI * R
  const frac = total > 0 ? Math.max(0, remaining / total) : 0
  const dash = frac * CIRC

  return (
    <svg width={72} height={72} className="rotate-[-90deg]">
      <circle cx={36} cy={36} r={R} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={6} />
      <circle
        cx={36} cy={36} r={R}
        fill="none"
        stroke={color}
        strokeWidth={6}
        strokeLinecap="round"
        strokeDasharray={`${dash} ${CIRC}`}
        style={{ transition: 'stroke-dasharray 0.8s linear' }}
      />
    </svg>
  )
}

/* ── Signal lamp ── */
function SignalLamp({ isGreen }) {
  return (
    <div className="flex flex-col gap-1.5 items-center">
      {/* Red lamp */}
      <div className={`w-7 h-7 rounded-full border-2 transition-all duration-500 ${
        !isGreen
          ? 'bg-red-500 border-red-400 shadow-[0_0_14px_rgba(239,68,68,0.8)]'
          : 'bg-red-950 border-red-900/40'
      }`} />
      {/* Yellow (always dim) */}
      <div className="w-7 h-7 rounded-full bg-yellow-950 border-2 border-yellow-900/30" />
      {/* Green lamp */}
      <div className={`w-7 h-7 rounded-full border-2 transition-all duration-500 ${
        isGreen
          ? 'bg-green-400 border-green-300 shadow-[0_0_14px_rgba(74,222,128,0.8)]'
          : 'bg-green-950 border-green-900/40'
      }`} />
    </div>
  )
}

/* ── Lane signal card ── */
function LaneCard({ lane, count, isGreen, remaining, greenDur }) {
  const laneColor  = lane === 'A' ? '#00dbb4' : '#ff9b2a'
  const statusText = isGreen ? `GO  —  ${remaining}s` : 'STOP'
  const statusClr  = isGreen ? 'text-green-400' : 'text-red-400'
  const ringClr    = isGreen ? '#4ade80' : '#ef4444'

  return (
    <motion.div
      layout
      className={`relative glass rounded-2xl border p-5 flex flex-col gap-4 overflow-hidden
                  transition-all duration-500 ${
        isGreen
          ? 'border-green-500/40 shadow-[0_0_30px_rgba(74,222,128,0.12)]'
          : 'border-white/[0.07]'
      }`}
    >
      {/* Glow strip at top */}
      <div
        className="absolute top-0 left-0 right-0 h-0.5 rounded-t-2xl transition-opacity duration-500"
        style={{ background: laneColor, opacity: isGreen ? 0.9 : 0.25 }}
      />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
               style={{ background: `${laneColor}18`, border: `1px solid ${laneColor}40` }}>
            <TrafficCone className="w-4 h-4" style={{ color: laneColor }} />
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-widest font-medium">Zone</p>
            <p className="text-base font-bold text-white" style={{ color: laneColor }}>
              Lane {lane}
            </p>
          </div>
        </div>

        {/* Traffic light */}
        <div className="bg-gray-900/80 rounded-xl p-2 border border-white/[0.06]">
          <SignalLamp isGreen={isGreen} />
        </div>
      </div>

      {/* Status + countdown */}
      <div className="flex items-center justify-between">
        <div>
          <p className={`text-xl font-bold tracking-tight ${statusClr}`}>
            {statusText}
          </p>
          <p className="text-gray-500 text-xs mt-0.5">
            {isGreen ? `${greenDur}s cycle` : 'waiting for green'}
          </p>
        </div>

        {/* Ring only shown when green */}
        <AnimatePresence>
          {isGreen && (
            <motion.div
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.6 }}
              className="relative flex items-center justify-center"
            >
              <CountdownRing remaining={remaining} total={greenDur} color={ringClr} />
              <div className="absolute flex flex-col items-center">
                <span className="text-base font-bold text-white leading-none">{remaining}</span>
                <span className="text-[9px] text-gray-500 leading-none">sec</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Vehicle count */}
      <div className="border-t border-white/[0.05] pt-3 flex items-center gap-2">
        <Car className="w-4 h-4 text-gray-500 flex-shrink-0" />
        <span className="text-gray-400 text-sm">
          <span className="text-white font-semibold">{count}</span> vehicle{count !== 1 ? 's' : ''} detected
        </span>
      </div>
    </motion.div>
  )
}

/* ── Main page ── */
export default function TrafficSignal() {
  const [status, setStatus] = useState({
    lane_a_count: 0,
    lane_b_count: 0,
    green_lane:   null,
    remaining:    0,
    green_dur:    0,
  })
  const [uploading,   setUploading]   = useState(false)
  const [sourceLabel, setSourceLabel] = useState('Live Webcam')
  const [error,       setError]       = useState(null)
  const fileRef = useRef(null)
  const pollRef = useRef(null)

  /* ── Poll signal status ── */
  const fetchStatus = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API}/signal_status`)
      setStatus(data)
      setError(null)
    } catch {
      setError('Cannot reach backend (http://localhost:9000)')
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    pollRef.current = setInterval(fetchStatus, POLL_MS)
    return () => clearInterval(pollRef.current)
  }, [fetchStatus])

  /* ── Source controls ── */
  const switchWebcam = async () => {
    try {
      await axios.post(`${API}/signal_use_webcam`)
      setSourceLabel('Live Webcam')
    } catch {
      setError('Failed to switch to webcam.')
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    const form = new FormData()
    form.append('video', file)
    try {
      const { data } = await axios.post(`${API}/signal_upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      if (data.success) setSourceLabel(data.filename)
      else setError(data.error || 'Upload failed.')
    } catch {
      setError('Upload failed.')
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  const { lane_a_count, lane_b_count, green_lane, remaining, green_dur } = status
  const totalVehicles = lane_a_count + lane_b_count

  /* timing rule display */
  const TIMING_RULES = [
    { range: '0 – 5',   sec: '10s', active: totalVehicles <= 5 },
    { range: '6 – 10',  sec: '20s', active: totalVehicles > 5  && totalVehicles <= 10 },
    { range: '11 – 15', sec: '30s', active: totalVehicles > 10 && totalVehicles <= 15 },
    { range: '16 – 20', sec: '40s', active: totalVehicles > 15 && totalVehicles <= 20 },
    { range: '> 20',    sec: '50s', active: totalVehicles > 20 },
  ]

  return (
    <PageWrapper>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* ── Page header ── */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
              <TrafficCone className="w-6 h-6 text-yellow-400" />
              Smart Traffic Signal
            </h1>
            <p className="text-gray-500 text-sm mt-0.5">
              YOLO vehicle detection · Zone-based density control · Adaptive timing
            </p>
          </div>

          {/* Source controls */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-500 bg-white/[0.04] border border-white/[0.07]
                             rounded-lg px-3 py-1.5 flex items-center gap-1.5">
              <Circle className="w-2 h-2 fill-green-400 text-green-400 animate-pulse" />
              {sourceLabel}
            </span>
            <button
              onClick={switchWebcam}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
                         bg-white/[0.05] border border-white/[0.08] text-gray-300
                         hover:bg-white/[0.10] hover:text-white transition-colors duration-150"
            >
              <Camera className="w-4 h-4" /> Webcam
            </button>
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
                         bg-yellow-400/10 border border-yellow-400/30 text-yellow-400
                         hover:bg-yellow-400/20 transition-colors duration-150 disabled:opacity-50"
            >
              {uploading
                ? <RefreshCw className="w-4 h-4 animate-spin" />
                : <Upload className="w-4 h-4" />
              }
              {uploading ? 'Uploading…' : 'Upload Video'}
            </button>
            <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
          </div>
        </div>

        {/* ── Error banner ── */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2 bg-red-500/10 border border-red-500/30
                         rounded-xl px-4 py-3 text-red-400 text-sm"
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Main grid: video left, controls right ── */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-6">

          {/* Video feed */}
          <div className="glass rounded-2xl border border-white/[0.07] overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/[0.06]">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="w-2 h-2 rounded-full bg-yellow-400" />
                <span className="w-2 h-2 rounded-full bg-green-400" />
              </div>
              <span className="text-xs text-gray-500 font-mono">
                signal_video • 854 × 480 • MJPEG
              </span>
              <div className="flex items-center gap-1.5 text-xs text-green-400">
                <Circle className="w-1.5 h-1.5 fill-green-400 animate-pulse" /> LIVE
              </div>
            </div>
            <img
              src={`${API}/signal_video`}
              alt="Traffic signal feed"
              className="w-full block"
              style={{ aspectRatio: '854/480', objectFit: 'cover', background: '#0a0a0f' }}
            />
          </div>

          {/* Right column */}
          <div className="flex flex-col gap-4">

            {/* Lane A card */}
            <LaneCard
              lane="A"
              count={lane_a_count}
              isGreen={green_lane === 'A'}
              remaining={remaining}
              greenDur={green_dur}
            />

            {/* Lane B card */}
            <LaneCard
              lane="B"
              count={lane_b_count}
              isGreen={green_lane === 'B'}
              remaining={remaining}
              greenDur={green_dur}
            />

            {/* Timing rules card */}
            <div className="glass rounded-2xl border border-white/[0.07] p-4">
              <p className="text-xs text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                <Timer className="w-3.5 h-3.5" /> Timing Rules
              </p>
              <div className="space-y-2">
                {TIMING_RULES.map(({ range, sec, active }) => (
                  <div
                    key={range}
                    className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm
                                transition-all duration-300 ${
                      active
                        ? 'bg-yellow-400/10 border border-yellow-400/30 text-yellow-300'
                        : 'bg-white/[0.03] border border-transparent text-gray-500'
                    }`}
                  >
                    <span>{range} vehicles</span>
                    <span className="font-mono font-bold">{sec}</span>
                    {active && <CheckCircle2 className="w-3.5 h-3.5 text-yellow-400" />}
                  </div>
                ))}
              </div>
            </div>

            {/* Stats summary */}
            <div className="glass rounded-2xl border border-white/[0.07] p-4">
              <p className="text-xs text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                <Car className="w-3.5 h-3.5" /> Intersection Summary
              </p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Lane A',    value: lane_a_count, color: '#00dbb4' },
                  { label: 'Lane B',    value: lane_b_count, color: '#ff9b2a' },
                  { label: 'Total',     value: totalVehicles, color: '#a78bfa' },
                  { label: 'Green Lane', value: green_lane ? `Lane ${green_lane}` : '—', color: '#4ade80' },
                ].map(({ label, value, color }) => (
                  <div key={label}
                       className="bg-white/[0.03] rounded-xl p-3 border border-white/[0.05]">
                    <p className="text-gray-500 text-xs">{label}</p>
                    <p className="text-lg font-bold mt-0.5" style={{ color }}>{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── Zone legend ── */}
        <div className="glass rounded-2xl border border-white/[0.07] p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Zone Legend</p>
          <div className="flex flex-wrap gap-6">
            {[
              { color: '#00dbb4', label: 'Lane A — Top road region',   desc: 'Polygon drawn in teal on the video feed' },
              { color: '#ff9b2a', label: 'Lane B — Right road region',  desc: 'Polygon drawn in amber on the video feed' },
              { color: '#a0a0a0', label: 'Unassigned vehicles',         desc: 'Outside both zones — counted but not assigned' },
            ].map(({ color, label, desc }) => (
              <div key={label} className="flex items-start gap-2.5">
                <div className="w-3 h-3 rounded-sm mt-0.5 flex-shrink-0"
                     style={{ background: color }} />
                <div>
                  <p className="text-sm font-medium text-white">{label}</p>
                  <p className="text-xs text-gray-500">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </PageWrapper>
  )
}
