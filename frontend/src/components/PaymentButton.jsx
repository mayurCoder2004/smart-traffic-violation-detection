import { useState } from 'react'
import { createOrder, verifyPayment } from '../api'

const TYPE_LABEL = {
  helmet:    'No Helmet',
  triple:    'Triple Riding',
  overspeed: 'Overspeed',
}

export default function PaymentButton({ violation, onSuccess }) {
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
        name:        'TrafficGuard',
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
      <span className="inline-flex items-center gap-1 text-green-400 text-sm font-semibold">
        <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
        Paid
      </span>
    )
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={handlePay}
        disabled={loading}
        className="bg-yellow-400 text-gray-900 text-sm font-bold px-4 py-2 rounded-lg
                   hover:bg-yellow-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? 'Opening...' : `Pay ₹${violation.fine_amount}`}
      </button>
      {error && <p className="text-red-400 text-xs max-w-[180px] text-right">{error}</p>}
    </div>
  )
}
