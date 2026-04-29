import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import {
  Shield, Search, RefreshCw, CheckCircle2, XCircle,
  AlertCircle, FileText, Check, Car, Clock,
  ScanLine, BadgeAlert,
} from 'lucide-react'

const API = 'http://localhost:9000'

// Quick-fill plates are fetched from the server on mount so they always
// reflect whichever dataset the backend has loaded (5-entry or 1000-record).

const VIOLATION_OPTIONS = [
  { key: 'helmet',    label: 'No Helmet',     fine: 500  },
  { key: 'triple',    label: 'Triple Riding',  fine: 1000 },
  { key: 'overspeed', label: 'Overspeed',      fine: 1000 },
]

const DOC_META = {
  rc:        { label: 'RC (Registration)',    fine: 0    },
  insurance: { label: 'Insurance',            fine: 1000 },
  puc:       { label: 'PUC Certificate',      fine: 500  },
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

function StatusPill({ paid }) {
  return (
    <span className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded-full ${
      paid
        ? 'bg-green-500/20 text-green-400 border border-green-500/30'
        : 'bg-red-500/20  text-red-400  border border-red-500/30'
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${paid ? 'bg-green-400' : 'bg-red-400'}`} />
      {paid ? 'PAID' : 'UNPAID'}
    </span>
  )
}

export default function ChallanPolice() {
  const [plate,      setPlate]      = useState('')
  const [scanning,   setScanning]   = useState(false)
  const [scanned,    setScanned]    = useState(null)
  const [scanError,  setScanError]  = useState(null)
  const [selected,   setSelected]   = useState([])
  const [issuing,    setIssuing]    = useState(false)
  const [issued,     setIssued]     = useState(null)
  const [challans,   setChallans]   = useState([])
  const [quickPlates, setQuickPlates] = useState([])
  const inputRef = useRef(null)

  /* ── Load quick-fill plates from server on mount ── */
  useEffect(() => {
    axios.get(`${API}/sample-plates?n=8`)
      .then(({ data }) => setQuickPlates(data))
      .catch(() => {})
  }, [])

  /* ── Poll all challans ── */
  const loadChallans = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API}/all-challans`)
      setChallans(data.challans)
    } catch {}
  }, [])

  useEffect(() => {
    loadChallans()
    const id = setInterval(loadChallans, 2500)
    return () => clearInterval(id)
  }, [loadChallans])

  /* ── Scan ── */
  const handleScan = async (overridePlate) => {
    const p = (overridePlate ?? plate).trim().toUpperCase()
    if (!p) return
    setPlate(p)
    setScanning(true)
    setScanError(null)
    setScanned(null)
    setSelected([])
    setIssued(null)
    try {
      const { data } = await axios.get(`${API}/scan/${p}`)
      setScanned(data)
    } catch (e) {
      setScanError(e.response?.data?.message ?? 'Vehicle record not found.')
    } finally {
      setScanning(false)
    }
  }

  /* ── Toggle violation checkbox ── */
  const toggle = (key) =>
    setSelected(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key])

  /* ── Live fine ── */
  const liveFine = (() => {
    let f = 0
    if (scanned?.documents) {
      const d = scanned.documents
      if (!d.insurance) f += 1000
      if (!d.puc)       f += 500
    }
    selected.forEach(k => { f += VIOLATION_OPTIONS.find(o => o.key === k)?.fine ?? 0 })
    return f
  })()

  /* ── Issue challan ── */
  const handleIssue = async () => {
    setIssuing(true)
    try {
      const { data } = await axios.post(`${API}/create-challan`, {
        plate:               plate.toUpperCase(),
        selected_violations: selected,
      })
      setIssued({ ...data.challan, skipped: data.skipped ?? [] })
      setScanned(null)
      setSelected([])
      loadChallans()
    } catch (e) {
      setScanError(e.response?.data?.error ?? 'Failed to issue challan.')
    } finally {
      setIssuing(false)
    }
  }

  const paidCount   = challans.filter(c => c.status === 'PAID').length
  const unpaidCount = challans.length - paidCount

  return (
    <PageWrapper>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* ── Header ── */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-yellow-400/10 border border-yellow-400/30
                              flex items-center justify-center">
                <Shield className="w-4 h-4 text-yellow-400" fill="currentColor" />
              </div>
              Police Scanner
            </h1>
            <p className="text-gray-500 text-sm mt-0.5">
              Scan plate · Verify documents · Issue challan
            </p>
          </div>

          {/* Stats strip */}
          <div className="flex items-center gap-3">
            {[
              { label: 'Total',  value: challans.length, color: 'text-white' },
              { label: 'Unpaid', value: unpaidCount,      color: 'text-red-400' },
              { label: 'Paid',   value: paidCount,        color: 'text-green-400' },
            ].map(({ label, value, color }) => (
              <div key={label}
                   className="glass border border-white/[0.07] rounded-xl px-4 py-2 text-center min-w-[72px]">
                <p className={`text-xl font-bold ${color}`}>{value}</p>
                <p className="text-gray-600 text-[11px]">{label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── Main grid ── */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-6 items-start">

          {/* ── LEFT: Scanner + form ── */}
          <div className="space-y-4">

            {/* Scan input */}
            <div className="glass rounded-2xl border border-white/[0.07] p-5">
              <p className="text-xs text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                <ScanLine className="w-3.5 h-3.5" /> Number Plate Scanner
              </p>

              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  value={plate}
                  onChange={e => setPlate(e.target.value.toUpperCase())}
                  onKeyDown={e => e.key === 'Enter' && handleScan()}
                  placeholder="e.g. KA01AB1234"
                  className="flex-1 bg-white/[0.04] border border-white/[0.10] rounded-xl px-4 py-2.5
                             text-white font-mono text-base placeholder-gray-600 outline-none
                             focus:border-yellow-400/50 focus:ring-1 focus:ring-yellow-400/20
                             transition-all duration-200"
                />
                <button
                  onClick={() => handleScan()}
                  disabled={scanning || !plate.trim()}
                  className="px-5 py-2.5 rounded-xl bg-yellow-400 text-gray-900 font-semibold text-sm
                             hover:bg-yellow-300 active:scale-95 disabled:opacity-40
                             transition-all duration-150 flex items-center gap-2"
                >
                  {scanning
                    ? <RefreshCw className="w-4 h-4 animate-spin" />
                    : <Search className="w-4 h-4" />}
                  {scanning ? 'Scanning…' : 'Scan'}
                </button>
              </div>

              {/* Quick-fill buttons — sourced live from /sample-plates */}
              {quickPlates.length > 0 && (
                <div className="mt-3 space-y-1.5">
                  <p className="text-[11px] text-gray-600 uppercase tracking-wider">
                    Simulated YOLO scan — quick fill:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {quickPlates.map(({ plate: p, vehicle, issues }) => (
                      <button
                        key={p}
                        onClick={() => handleScan(p)}
                        className="flex items-center gap-1.5 text-xs font-mono px-3 py-1.5 rounded-lg
                                   bg-white/[0.04] border border-white/[0.08] text-gray-400
                                   hover:text-yellow-400 hover:border-yellow-400/30 hover:bg-yellow-400/[0.05]
                                   transition-all duration-150"
                      >
                        <ScanLine className="w-3 h-3" />
                        {p}
                        <span className="text-gray-600 font-sans">
                          ({vehicle}{issues.length > 0 ? ` · no ${issues.join('/')}` : ' · clean'})
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Scan error */}
            <AnimatePresence>
              {scanError && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex items-center gap-2.5 bg-red-500/10 border border-red-500/30
                             rounded-xl px-4 py-3 text-red-400 text-sm"
                >
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {scanError}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Challan issued success */}
            <AnimatePresence>
              {issued && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  className="bg-green-500/10 border border-green-500/30 rounded-2xl p-5"
                >
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-green-400 font-bold text-lg">Challan Issued Successfully!</p>
                      <p className="text-gray-400 text-sm mt-0.5">
                        ID <span className="font-mono text-white">{issued.id}</span> ·{' '}
                        {issued.owner} · {issued.plate}
                      </p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {issued.violations.map((v, i) => (
                          <span key={i}
                                className="text-xs px-2.5 py-1 rounded-full bg-white/[0.06] border border-white/[0.08] text-gray-300">
                            {v.type} — ₹{v.fine}
                          </span>
                        ))}
                      </div>
                      {issued.skipped?.length > 0 && (
                        <p className="text-amber-300 text-xs mt-3">
                          Skipped duplicate within 24 hours: {issued.skipped.join(', ')}
                        </p>
                      )}
                      <p className="text-2xl font-bold font-mono text-red-400 mt-3">
                        Total: ₹{issued.fine}
                      </p>
                    </div>
                    <button
                      onClick={() => setIssued(null)}
                      className="text-gray-600 hover:text-gray-400 text-lg leading-none"
                    >✕</button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Vehicle info card */}
            <AnimatePresence>
              {scanned && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="glass rounded-2xl border border-white/[0.07] p-5 space-y-5"
                >
                  {/* Owner row */}
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[11px] text-gray-500 uppercase tracking-widest">Registered Owner</p>
                      <p className="text-white text-xl font-bold mt-0.5">{scanned.owner}</p>
                      <p className="text-gray-400 text-sm mt-0.5 flex items-center gap-1.5">
                        <Car className="w-3.5 h-3.5" />
                        {scanned.vehicle} ·
                        <span className="font-mono text-gray-300">{scanned.plate}</span>
                      </p>
                    </div>
                    <span className="flex-shrink-0 flex items-center gap-1.5 text-xs font-medium
                                     bg-green-500/10 border border-green-500/25 text-green-400
                                     rounded-lg px-3 py-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Registered
                    </span>
                  </div>

                  {/* Document grid */}
                  <div>
                    <p className="text-[11px] text-gray-500 uppercase tracking-widest mb-2.5">
                      Document Verification
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(scanned.documents).map(([doc, valid]) => {
                        const { label, fine } = DOC_META[doc]
                        return (
                          <div key={doc}
                               className={`flex items-center justify-between rounded-xl px-3.5 py-3 border
                                           transition-all duration-200 ${
                                 valid
                                   ? 'bg-green-500/[0.05] border-green-500/20'
                                   : 'bg-red-500/[0.08]  border-red-500/25'
                               }`}>
                            <div className="flex items-center gap-2">
                              {valid
                                ? <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0" />
                                : <XCircle      className="w-4 h-4 text-red-400   flex-shrink-0" />}
                              <span className="text-sm text-gray-300">{label}</span>
                            </div>
                            {!valid && fine > 0 && (
                              <span className="text-xs font-mono text-red-400 font-bold ml-1">+₹{fine}</span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* Violations */}
                  <div>
                    <p className="text-[11px] text-gray-500 uppercase tracking-widest mb-2.5">
                      Add Violations
                    </p>
                    <div className="space-y-2">
                      {VIOLATION_OPTIONS.map(({ key, label, fine }) => {
                        const on = selected.includes(key)
                        return (
                          <button
                            key={key}
                            onClick={() => toggle(key)}
                            className={`w-full flex items-center justify-between rounded-xl px-4 py-3 border
                                        text-left transition-all duration-150 group ${
                              on
                                ? 'bg-orange-500/10 border-orange-500/40'
                                : 'bg-white/[0.03] border-white/[0.07] hover:bg-white/[0.06] hover:border-white/[0.12]'
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              <div className={`w-4.5 h-4.5 w-[18px] h-[18px] rounded border-2 flex-shrink-0
                                              flex items-center justify-center transition-all duration-150 ${
                                on
                                  ? 'bg-orange-500 border-orange-400'
                                  : 'border-gray-600 group-hover:border-gray-500'
                              }`}>
                                {on && <Check className="w-2.5 h-2.5 text-white" />}
                              </div>
                              <span className={`text-sm font-medium ${on ? 'text-orange-300' : 'text-gray-300'}`}>
                                {label}
                              </span>
                            </div>
                            <span className={`text-sm font-mono font-bold ${on ? 'text-orange-400' : 'text-gray-600'}`}>
                              +₹{fine}
                            </span>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  {/* Live fine */}
                  <div className={`rounded-xl border px-4 py-3.5 flex items-center justify-between
                                   transition-all duration-300 ${
                    liveFine > 0
                      ? 'bg-red-500/[0.08] border-red-500/30'
                      : 'bg-white/[0.03]   border-white/[0.07]'
                  }`}>
                    <div>
                      <p className="text-gray-500 text-xs">Total Fine</p>
                      <p className={`text-3xl font-bold font-mono mt-0.5 transition-colors duration-200 ${
                        liveFine > 0 ? 'text-red-400' : 'text-gray-600'
                      }`}>
                        ₹{liveFine}
                      </p>
                    </div>
                    {liveFine === 0 && (
                      <p className="text-gray-600 text-sm">Select violations to calculate fine</p>
                    )}
                  </div>

                  {/* Issue button */}
                  <button
                    onClick={handleIssue}
                    disabled={issuing || liveFine === 0}
                    className="w-full py-3.5 rounded-xl bg-red-500 hover:bg-red-400 active:scale-[0.99]
                               text-white font-bold text-sm disabled:opacity-40
                               transition-all duration-150 flex items-center justify-center gap-2"
                  >
                    {issuing
                      ? <RefreshCw className="w-4 h-4 animate-spin" />
                      : <BadgeAlert className="w-4 h-4" />}
                    {issuing ? 'Issuing Challan…' : 'Issue Challan'}
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ── RIGHT: All challans ── */}
          <div className="glass rounded-2xl border border-white/[0.07] p-5">
            <div className="flex items-center justify-between mb-4">
              <p className="text-xs text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" /> All Challans
              </p>
              <span className="text-[11px] text-gray-600 bg-white/[0.04] rounded-full px-2.5 py-0.5">
                {challans.length} total
              </span>
            </div>

            {challans.length === 0 ? (
              <div className="text-center py-14 text-gray-600">
                <FileText className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-sm">No challans issued yet.</p>
                <p className="text-xs mt-1">Scan a plate to get started.</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[680px] overflow-y-auto pr-0.5">
                {[...challans].reverse().map((c) => (
                  <motion.div
                    key={c.id}
                    initial={{ opacity: 0, x: 8 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`rounded-xl border p-3.5 transition-all duration-300 ${
                      c.status === 'PAID'
                        ? 'bg-green-500/[0.05] border-green-500/20'
                        : 'bg-red-500/[0.05]   border-red-500/20'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div>
                        <p className="font-mono text-sm font-bold text-white leading-tight">{c.plate}</p>
                        <p className="text-gray-500 text-xs mt-0.5">{c.owner} · {c.vehicle}</p>
                      </div>
                      <div className="flex flex-col items-end gap-1 flex-shrink-0">
                        <StatusPill paid={c.status === 'PAID'} />
                        <p className="text-[11px] text-gray-600 font-mono">{c.id}</p>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-1 mb-2">
                      {c.violations.map((v, i) => (
                        <span key={i}
                              className="text-[11px] px-2 py-0.5 rounded-full bg-white/[0.05]
                                         border border-white/[0.07] text-gray-400">
                          {v.type}
                        </span>
                      ))}
                    </div>

                    <div className="flex items-center justify-between">
                      <p className="text-gray-600 text-[11px] flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(c.timestamp).toLocaleString('en-IN', {
                          dateStyle: 'short', timeStyle: 'short',
                        })}
                      </p>
                      <p className={`font-mono font-bold text-sm ${
                        c.status === 'PAID' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        ₹{c.fine}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </PageWrapper>
  )
}
