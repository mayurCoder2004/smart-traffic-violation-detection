import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { getUserViolations } from '../api'
import PaymentButton from './PaymentButton'
import {
  User, Car, AlertTriangle, IndianRupee, CheckCircle2,
  Clock, Loader2, ClipboardList, AlertCircle,
} from 'lucide-react'

/* ── Data maps — unchanged from original ── */
const TYPE_LABEL = {
  helmet:    'No Helmet',
  triple:    'Triple Riding',
  overspeed: 'Overspeed',
}

const TYPE_CONFIG = {
  helmet:    { color: 'text-red-400',    bg: 'bg-red-500/10',    border: 'border-red-500/20',    dot: 'bg-red-400' },
  triple:    { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/20', dot: 'bg-orange-400' },
  overspeed: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/20', dot: 'bg-yellow-400' },
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

function AnimatedNumber({ value, prefix = '' }) {
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    const end = Number(value)
    if (end === 0) { setDisplay(0); return }
    const step  = Math.max(1, Math.ceil(end / 28))
    let current = 0
    const timer = setInterval(() => {
      current += step
      if (current >= end) { setDisplay(end); clearInterval(timer) }
      else setDisplay(current)
    }, 28)
    return () => clearInterval(timer)
  }, [value])
  return <>{prefix}{display.toLocaleString('en-IN')}</>
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

const listStagger = {
  initial: {},
  animate: { transition: { staggerChildren: 0.07, delayChildren: 0.05 } },
}
const itemVariant = {
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

export default function UserDashboard() {
  /* ── All existing logic unchanged ── */
  const navigate          = useNavigate()
  const [user, setUser]   = useState(null)
  const [violations, setViolations] = useState([])
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState('')

  useEffect(() => {
    const stored = sessionStorage.getItem('user')
    if (!stored) {
      navigate('/login')
      return
    }
    const u = JSON.parse(stored)
    setUser(u)

    getUserViolations(u.id)
      .then((res) => setViolations(res.data))
      .catch(() => setError('Failed to load violations.'))
      .finally(() => setLoading(false))
  }, [navigate])

  const handlePaymentSuccess = (violationId) => {
    setViolations((prev) =>
      prev.map((v) => (v.id === violationId ? { ...v, status: 'PAID' } : v))
    )
  }

  const pendingCount = violations.filter((v) => v.status === 'PENDING').length
  const totalFine    = violations
    .filter((v) => v.status === 'PENDING')
    .reduce((s, v) => s + v.fine_amount, 0)

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div className="w-11 h-11 rounded-xl bg-yellow-400/10 border border-yellow-400/20
                        flex items-center justify-center">
          <Loader2 className="w-5 h-5 text-yellow-400 animate-spin" />
        </div>
        <p className="text-gray-500 text-sm">Loading your challans...</p>
      </div>
    )
  }

  return (
    <PageWrapper>
      <div className="max-w-4xl mx-auto px-3 sm:px-4 py-5 sm:py-8">

        {/* ── Profile card ── */}
        {user && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="glass rounded-2xl p-4 sm:p-6 mb-5 sm:mb-6
                       hover:shadow-[0_0_35px_rgba(240,192,64,0.07)] transition-shadow duration-400"
          >
            {/* Stack vertically on xs, row on sm+ */}
            <div className="flex flex-col xs:flex-row sm:flex-row items-start sm:items-center
                            justify-between gap-4">
              <div className="flex items-center gap-3 sm:gap-4">
                <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-xl bg-yellow-400/10 border border-yellow-400/25
                                flex items-center justify-center text-yellow-400 font-bold text-lg sm:text-xl
                                flex-shrink-0">
                  {user.name?.charAt(0).toUpperCase() || <User className="w-5 h-5" />}
                </div>
                <div className="min-w-0">
                  <p className="text-gray-500 text-[10px] font-semibold uppercase tracking-widest mb-0.5">
                    Registered Driver
                  </p>
                  <h1 className="text-lg sm:text-xl font-bold text-white truncate">{user.name}</h1>
                  <p className="text-gray-400 text-sm mt-0.5">{user.phone}</p>
                </div>
              </div>

              {/* License plate badge */}
              <div className="flex flex-row sm:flex-col items-center sm:items-end gap-2 sm:gap-1.5 flex-shrink-0">
                <p className="text-gray-500 text-[10px] font-semibold uppercase tracking-widest
                               flex items-center gap-1 sm:hidden">
                  <Car className="w-3 h-3" />
                </p>
                <p className="text-gray-500 text-[10px] font-semibold uppercase tracking-widest
                               items-center gap-1 hidden sm:flex">
                  <Car className="w-3 h-3" /> Vehicle Plate
                </p>
                <div className="font-plate text-base sm:text-lg text-yellow-400
                                border border-yellow-400/30 rounded-lg px-3 sm:px-4 py-1.5 sm:py-2
                                bg-yellow-400/[0.06]">
                  {user.license_plate}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* ── Stats row ── */}
        <div className="grid grid-cols-2 gap-3 sm:gap-4 mb-6 sm:mb-8">
          {[
            {
              label:  'Pending Challans',
              value:  pendingCount,
              prefix: '',
              color:  'text-red-400',
              bg:     'bg-red-500/10',
              border: 'border-red-500/20',
              Icon:   AlertTriangle,
              glow:   'hover:shadow-[0_0_28px_rgba(239,68,68,0.12)]',
              delay:  0,
            },
            {
              label:  'Total Pending Fine',
              value:  totalFine,
              prefix: '₹',
              color:  'text-yellow-400',
              bg:     'bg-yellow-500/10',
              border: 'border-yellow-500/20',
              Icon:   IndianRupee,
              glow:   'hover:shadow-[0_0_28px_rgba(240,192,64,0.12)]',
              delay:  0.08,
            },
          ].map(({ label, value, prefix, color, bg, border, Icon, glow, delay }) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay }}
              className={`glass rounded-xl p-4 sm:p-5 ${glow} transition-shadow duration-300`}
            >
              <div className="flex items-center justify-between mb-2 sm:mb-3">
                <p className="text-gray-500 text-[10px] font-semibold uppercase tracking-widest
                               leading-tight pr-1">{label}</p>
                <div className={`w-6 h-6 sm:w-7 sm:h-7 rounded-lg ${bg} border ${border}
                                  flex items-center justify-center flex-shrink-0`}>
                  <Icon className={`w-3 h-3 sm:w-3.5 sm:h-3.5 ${color}`} />
                </div>
              </div>
              <p className={`text-3xl sm:text-4xl font-bold ${color}`}>
                <AnimatedNumber value={value} prefix={prefix} />
              </p>
            </motion.div>
          ))}
        </div>

        {/* ── Violations header ── */}
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <h2 className="text-xs font-semibold text-gray-300 flex items-center gap-2 uppercase tracking-widest">
            <ClipboardList className="w-4 h-4 text-gray-500" />
            Violation History
          </h2>
          <span className="text-xs text-gray-600 bg-white/[0.04] border border-white/[0.06]
                           rounded-full px-3 py-1">
            {violations.length} total
          </span>
        </div>

        {/* Error banner */}
        {error && (
          <div className="flex items-center gap-2 text-red-400 text-sm
                          bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 mb-4">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Empty state */}
        {violations.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-xl p-10 sm:p-12 text-center"
          >
            <div className="w-12 h-12 rounded-xl bg-green-500/10 border border-green-500/20
                            flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="w-6 h-6 text-green-400" />
            </div>
            <p className="text-gray-300 font-semibold">No violations on record</p>
            <p className="text-gray-600 text-sm mt-1">Drive safe and stay compliant!</p>
          </motion.div>
        ) : (
          <motion.div
            variants={listStagger}
            initial="initial"
            animate="animate"
            className="space-y-3"
          >
            {violations.map((v) => {
              const cfg = TYPE_CONFIG[v.violation_type] || {
                color: 'text-gray-400', bg: 'bg-gray-500/10', border: 'border-gray-500/20', dot: 'bg-gray-400',
              }
              return (
                <motion.div
                  key={v.id}
                  variants={itemVariant}
                  whileHover={{ y: -2 }}
                  /* Stack on xs, row on sm+ */
                  className="glass rounded-xl p-4 sm:p-5
                             flex flex-col xs:flex-row sm:flex-row
                             items-start sm:items-center justify-between gap-3 sm:gap-4
                             hover:border-white/[0.13] hover:shadow-[0_6px_24px_rgba(0,0,0,0.45)]
                             transition-all duration-200 cursor-default"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <span className={`inline-flex items-center gap-1.5 text-xs font-semibold
                                        px-2.5 py-1 rounded-md border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
                        {TYPE_LABEL[v.violation_type] || v.violation_type}
                      </span>
                      <span className={`text-xs px-2.5 py-1 rounded-md font-semibold border ${
                        v.status === 'PAID'
                          ? 'text-green-400 bg-green-500/10 border-green-500/20'
                          : 'text-red-400 bg-red-500/10 border-red-500/20'
                      }`}>
                        {v.status === 'PAID' ? '✓ PAID' : '⏳ PENDING'}
                      </span>
                    </div>
                    <p className="flex items-center gap-1.5 text-gray-600 text-xs mb-1.5">
                      <Clock className="w-3 h-3 flex-shrink-0" />
                      {formatDate(v.created_at)}
                    </p>
                    <p className="text-yellow-400 font-bold text-lg sm:text-xl">₹{v.fine_amount}</p>
                  </div>

                  {/* Payment button — flush right on sm+, full-width on xs */}
                  <div className="w-full xs:w-auto sm:w-auto flex-shrink-0">
                    <PaymentButton violation={v} onSuccess={handlePaymentSuccess} />
                  </div>
                </motion.div>
              )
            })}
          </motion.div>
        )}
      </div>
    </PageWrapper>
  )
}
