import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import {
  User, RefreshCw, CheckCircle2, XCircle,
  AlertCircle, CreditCard, Car, Clock, Receipt,
  BadgeCheck,
} from 'lucide-react'

const API = 'http://localhost:9000'

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

/* ── Single challan card shown to the user ── */
function ChallanCard({ challan, onPay, paying }) {
  const paid = challan.status === 'PAID'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl border p-5 transition-all duration-300 ${
        paid
          ? 'bg-green-500/[0.06] border-green-500/25 shadow-[0_0_18px_rgba(74,222,128,0.06)]'
          : 'bg-red-500/[0.06]   border-red-500/25   shadow-[0_0_18px_rgba(239,68,68,0.06)]'
      }`}
    >
      {/* Top row: ID + status */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <p className="font-mono text-xs text-gray-500">{challan.id}</p>
          <p className="text-white font-bold text-lg mt-0.5">
            {paid ? 'Paid' : 'Unpaid'} Challan
          </p>
          <p className="text-gray-500 text-xs flex items-center gap-1.5 mt-0.5">
            <Clock className="w-3 h-3" />
            {new Date(challan.timestamp).toLocaleString('en-IN', {
              dateStyle: 'medium', timeStyle: 'short',
            })}
          </p>
        </div>

        {/* Status badge */}
        <div className={`flex-shrink-0 px-3 py-1.5 rounded-xl text-sm font-bold border ${
          paid
            ? 'bg-green-500/15 border-green-500/30 text-green-400'
            : 'bg-red-500/15   border-red-500/30   text-red-400'
        }`}>
          {paid ? '✓ PAID' : '✗ UNPAID'}
        </div>
      </div>

      {/* Violations list */}
      <div className="space-y-2 mb-4">
        <p className="text-[11px] text-gray-500 uppercase tracking-widest">Violations</p>
        {challan.violations.map((v, i) => (
          <div key={i}
               className="flex items-center justify-between py-2 border-b border-white/[0.05] last:border-0">
            <div className="flex items-center gap-2">
              <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
              <span className="text-gray-300 text-sm">{v.type}</span>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                v.source === 'document'
                  ? 'bg-orange-500/15 text-orange-400'
                  : 'bg-red-500/15    text-red-400'
              }`}>
                {v.source === 'document' ? 'Doc' : 'Offence'}
              </span>
            </div>
            <span className="font-mono text-sm text-red-400 font-bold">₹{v.fine}</span>
          </div>
        ))}
      </div>

      {/* Fine + Pay */}
      <div className={`rounded-xl border px-4 py-3 flex items-center justify-between mb-4 ${
        paid
          ? 'bg-green-500/[0.06] border-green-500/20'
          : 'bg-red-500/[0.08]   border-red-500/20'
      }`}>
        <div>
          <p className="text-gray-500 text-xs">Total Fine</p>
          <p className={`text-2xl font-bold font-mono mt-0.5 ${
            paid ? 'text-green-400' : 'text-red-400'
          }`}>
            ₹{challan.fine}
          </p>
        </div>
        {paid && (
          <div className="flex items-center gap-1.5 text-green-400 text-sm font-medium">
            <CheckCircle2 className="w-5 h-5" />
            Payment Received
          </div>
        )}
      </div>

      {/* Pay button */}
      {!paid && (
        <button
          onClick={() => onPay(challan.id)}
          disabled={paying === challan.id}
          className="w-full py-3 rounded-xl bg-green-500 hover:bg-green-400 active:scale-[0.99]
                     text-white font-bold text-sm disabled:opacity-50
                     transition-all duration-150 flex items-center justify-center gap-2"
        >
          {paying === challan.id
            ? <RefreshCw className="w-4 h-4 animate-spin" />
            : <CreditCard className="w-4 h-4" />}
          {paying === challan.id ? 'Processing…' : 'Pay Now  ₹' + challan.fine}
        </button>
      )}
    </motion.div>
  )
}

export default function ChallanUser() {
  const [fetching, setFetching] = useState(false)
  const [profile,  setProfile]  = useState(null)   // { owner, vehicle, challans }
  const [fetchErr, setFetchErr] = useState(null)
  const [paying,   setPaying]   = useState(null)    // challan id being paid
  const [payErr,   setPayErr]   = useState(null)

  /* ── Fetch challans ── */
  const fetchUser = useCallback(async (plateNumber) => {
    const p = plateNumber?.trim().toUpperCase()
    if (!p) return
    setFetching(true)
    setFetchErr(null)
    setProfile(null)
    setPayErr(null)
    try {
      const { data } = await axios.get(`${API}/user/${p}`)
      setProfile(data)
    } catch (e) {
      setFetchErr(e.response?.data?.message ?? 'Could not load challans for your vehicle.')
    } finally {
      setFetching(false)
    }
  }, [])

  useEffect(() => {
    const stored = sessionStorage.getItem('user')
    if (!stored) {
      setFetchErr('Please log in with your vehicle plate to view challans.')
      return
    }

    try {
      const user = JSON.parse(stored)
      if (user?.license_plate) {
        fetchUser(user.license_plate)
      } else {
        setFetchErr('Your login session does not include a vehicle plate. Please log in again.')
      }
    } catch {
      sessionStorage.removeItem('user')
      setFetchErr('Your login session expired. Please log in again.')
    }
  }, [fetchUser])

  /* ── Pay challan ── */
  const handlePay = async (challanId) => {
    const plate = profile?.plate?.toUpperCase()
    if (!plate) {
      setPayErr('Vehicle plate missing. Please log in again.')
      return
    }
    if (!window.Razorpay) {
      setPayErr('Payment window could not load. Please refresh and try again.')
      return
    }

    setPaying(challanId)
    setPayErr(null)
    try {
      const { data: order } = await axios.post(`${API}/payments/create-scanner-order`, {
        challan_id: challanId,
        plate,
      })

      const rzp = new window.Razorpay({
        key: order.key_id,
        amount: order.amount * 100,
        currency: order.currency,
        name: 'VioLens',
        description: `Challan ${challanId}`,
        order_id: order.order_id,
        handler: async (response) => {
          try {
            await axios.post(`${API}/payments/verify-scanner`, {
              razorpay_order_id: response.razorpay_order_id,
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_signature: response.razorpay_signature,
              challan_id: challanId,
              plate,
            })
            await fetchUser(plate)
          } catch (e) {
            setPayErr(e.response?.data?.error ?? 'Payment verification failed. Please contact support.')
          } finally {
            setPaying(null)
          }
        },
        prefill: {
          name: profile.owner || '',
        },
        modal: {
          ondismiss: () => setPaying(null),
        },
        theme: { color: '#22c55e' },
      })

      rzp.on('payment.failed', (resp) => {
        setPayErr(resp.error?.description || 'Payment failed.')
        setPaying(null)
      })
      rzp.open()
    } catch (e) {
      const alreadyPaid = e.response?.status === 400 && e.response?.data?.error === 'Already paid'
      setPayErr(alreadyPaid ? 'This challan is already paid. Refreshing your challans.' : (e.response?.data?.error ?? 'Payment failed. Please try again.'))
      setPaying(null)
      if (alreadyPaid) fetchUser(plate)
    }
  }

  const unpaid = profile?.challans.filter(c => c.status === 'UNPAID') ?? []
  const paid   = profile?.challans.filter(c => c.status === 'PAID')   ?? []
  const totalDue = unpaid.reduce((s, c) => s + c.fine, 0)

  return (
    <PageWrapper>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* ── Header ── */}
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/30
                            flex items-center justify-center">
              <Receipt className="w-4 h-4 text-blue-400" />
            </div>
            My Challans
          </h1>
          <p className="text-gray-500 text-sm mt-0.5">
            View and pay challans for your logged-in vehicle
          </p>
        </div>

        {fetching && (
          <div className="glass rounded-2xl border border-white/[0.07] p-6 flex items-center gap-3">
            <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" />
            <p className="text-gray-400 text-sm">Loading challans for your vehicle...</p>
          </div>
        )}

        {/* Fetch error */}
        <AnimatePresence>
          {fetchErr && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2.5 bg-red-500/10 border border-red-500/30
                         rounded-xl px-4 py-3 text-red-400 text-sm"
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {fetchErr}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Pay error */}
        <AnimatePresence>
          {payErr && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2.5 bg-orange-500/10 border border-orange-500/30
                         rounded-xl px-4 py-3 text-orange-400 text-sm"
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {payErr}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Profile + challans */}
        <AnimatePresence>
          {profile && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-5"
            >
              {/* Profile card */}
              <div className="glass rounded-2xl border border-white/[0.07] p-5
                              flex flex-col sm:flex-row sm:items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/25
                                flex items-center justify-center flex-shrink-0">
                  <User className="w-6 h-6 text-blue-400" />
                </div>
                <div className="flex-1">
                  <p className="text-[11px] text-gray-500 uppercase tracking-widest mb-1 flex items-center gap-1.5">
                    <BadgeCheck className="w-3.5 h-3.5" />
                    Logged-in vehicle
                  </p>
                  <p className="text-white font-bold text-lg">{profile.owner}</p>
                  <p className="text-gray-400 text-sm flex items-center gap-1.5 mt-0.5">
                    <Car className="w-3.5 h-3.5" />
                    {profile.vehicle} ·
                    <span className="font-mono text-gray-300">{profile.plate}</span>
                  </p>
                </div>

                {/* Summary stats */}
                <div className="flex items-center gap-4 flex-wrap">
                  <div className="text-center">
                    <p className="text-xl font-bold text-white">{profile.challans.length}</p>
                    <p className="text-gray-600 text-xs">Total</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xl font-bold text-red-400">{unpaid.length}</p>
                    <p className="text-gray-600 text-xs">Unpaid</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xl font-bold text-green-400">{paid.length}</p>
                    <p className="text-gray-600 text-xs">Paid</p>
                  </div>
                  {totalDue > 0 && (
                    <div className="bg-red-500/10 border border-red-500/25 rounded-xl px-4 py-2">
                      <p className="text-[11px] text-gray-500">Amount Due</p>
                      <p className="text-xl font-bold font-mono text-red-400">₹{totalDue}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* No challans */}
              {profile.challans.length === 0 && (
                <div className="glass rounded-2xl border border-white/[0.07] p-10 text-center">
                  <CheckCircle2 className="w-10 h-10 text-green-400 mx-auto mb-3" />
                  <p className="text-white font-semibold text-lg">No Challans!</p>
                  <p className="text-gray-500 text-sm mt-1">
                    No challans have been issued for this vehicle. Drive safe!
                  </p>
                </div>
              )}

              {/* Unpaid challans first */}
              {unpaid.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                    <XCircle className="w-3.5 h-3.5 text-red-400" />
                    Pending Payment ({unpaid.length})
                  </p>
                  <div className="space-y-4">
                    {unpaid.map(c => (
                      <ChallanCard
                        key={c.id}
                        challan={c}
                        onPay={handlePay}
                        paying={paying}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Paid challans */}
              {paid.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                    Payment History ({paid.length})
                  </p>
                  <div className="space-y-4">
                    {paid.map(c => (
                      <ChallanCard
                        key={c.id}
                        challan={c}
                        onPay={handlePay}
                        paying={paying}
                      />
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </PageWrapper>
  )
}
