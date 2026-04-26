import { useState } from 'react'
import { motion } from 'framer-motion'
import { createOrder, verifyPayment } from '../api'
import { CheckCircle2, Loader2, CreditCard, AlertCircle } from 'lucide-react'

/* ── Data map — unchanged from original ── */
const TYPE_LABEL = {
  helmet:    'No Helmet',
  triple:    'Triple Riding',
  overspeed: 'Overspeed',
}

export default function PaymentButton({ violation, onSuccess }) {
  /* ── All existing logic unchanged ── */
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const handlePay = async () => {
    setLoading(true)
    setError('')
    try {
      const { data: order } = await createOrder(violation.id)

      const options = {
        key:         order.key_id,
        amount:      order.amount * 100,
        currency:    order.currency,
        name:        'VioLens',
        description: `Challan: ${TYPE_LABEL[violation.violation_type] || violation.violation_type}`,
        order_id:    order.order_id,
        handler: async (response) => {
          try {
            await verifyPayment({
              razorpay_order_id:   response.razorpay_order_id,
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_signature:  response.razorpay_signature,
              violation_id:        violation.id,
            })
            onSuccess(violation.id)
          } catch {
            setError('Payment verification failed. Contact support.')
          }
        },
        prefill: {
          name:    violation.user?.name  || '',
          contact: violation.user?.phone || '',
        },
        theme: { color: '#f0c040' },
      }

      const rzp = new window.Razorpay(options)
      rzp.on('payment.failed', (resp) => {
        setError(resp.error?.description || 'Payment failed.')
        setLoading(false)
      })
      rzp.open()
    } catch (err) {
      setError(err.response?.data?.error || 'Could not initiate payment.')
    } finally {
      setLoading(false)
    }
  }

  if (violation.status === 'PAID') {
    return (
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="flex items-center gap-1.5 text-green-400 text-sm font-semibold
                   bg-green-500/10 border border-green-500/20 rounded-xl px-3 py-2
                   flex-shrink-0"
      >
        <CheckCircle2 className="w-4 h-4" />
        <span>Paid</span>
      </motion.div>
    )
  }

  return (
    <div className="flex flex-col items-stretch xs:items-end sm:items-end gap-1.5 w-full xs:w-auto sm:w-auto">
      {/* Full width on xs (stacked layout), auto width on sm+ */}
      <button
        onClick={handlePay}
        disabled={loading}
        className="relative overflow-hidden flex items-center justify-center gap-2
                   w-full xs:w-auto sm:w-auto
                   bg-yellow-400 text-gray-900 text-sm font-bold px-4 py-2.5 rounded-xl
                   hover:bg-yellow-300 disabled:opacity-40 disabled:cursor-not-allowed
                   transition-all duration-200
                   hover:shadow-[0_0_22px_rgba(240,192,64,0.4)]
                   active:scale-[0.97] group"
      >
        <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full
                        transition-transform duration-500 bg-gradient-to-r
                        from-transparent via-white/25 to-transparent pointer-events-none" />
        {loading
          ? <><Loader2 className="w-4 h-4 animate-spin relative z-10" />
              <span className="relative z-10">Opening…</span></>
          : <><CreditCard className="w-4 h-4 relative z-10" />
              <span className="relative z-10">Pay ₹{violation.fine_amount}</span></>
        }
      </button>

      {error && (
        <motion.p
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-1 text-red-400 text-xs"
        >
          <AlertCircle className="w-3 h-3 flex-shrink-0" />
          {error}
        </motion.p>
      )}
    </div>
  )
}
