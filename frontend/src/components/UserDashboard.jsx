import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getUserViolations } from '../api'
import PaymentButton from './PaymentButton'

const TYPE_LABEL = {
  helmet:    'No Helmet',
  triple:    'Triple Riding',
  overspeed: 'Overspeed',
}

const TYPE_COLOR = {
  helmet:    'text-red-400 bg-red-950 border-red-800',
  triple:    'text-orange-400 bg-orange-950 border-orange-800',
  overspeed: 'text-yellow-400 bg-yellow-950 border-yellow-800',
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

export default function UserDashboard() {
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
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-gray-500">Loading your challans...</div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Profile card */}
      {user && (
        <div className="bg-[#141414] border border-gray-800 rounded-2xl p-6 mb-6 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-gray-500 text-sm mb-1">Registered Driver</p>
            <h1 className="text-2xl font-bold text-white">{user.name}</h1>
            <p className="text-gray-400 mt-1">{user.phone}</p>
          </div>
          <div className="text-right">
            <p className="text-gray-500 text-xs mb-1">Vehicle Plate</p>
            <p className="font-mono text-xl font-bold text-yellow-400 tracking-widest border border-yellow-600 rounded-lg px-4 py-2 bg-yellow-950">
              {user.license_plate}
            </p>
          </div>
        </div>
      )}

      {/* Summary row */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-[#141414] border border-gray-800 rounded-xl p-4">
          <p className="text-gray-500 text-xs mb-1">Pending Challans</p>
          <p className="text-3xl font-bold text-red-400">{pendingCount}</p>
        </div>
        <div className="bg-[#141414] border border-gray-800 rounded-xl p-4">
          <p className="text-gray-500 text-xs mb-1">Total Pending Fine</p>
          <p className="text-3xl font-bold text-yellow-400">₹{totalFine}</p>
        </div>
      </div>

      {/* Violations list */}
      <h2 className="text-lg font-semibold text-gray-300 mb-4">Violation History</h2>

      {error && (
        <p className="text-red-400 bg-red-950 border border-red-800 rounded-lg px-4 py-3 mb-4">
          {error}
        </p>
      )}

      {violations.length === 0 ? (
        <div className="bg-[#141414] border border-gray-800 rounded-xl p-10 text-center">
          <p className="text-gray-500">No violations on record. Drive safe!</p>
        </div>
      ) : (
        <div className="space-y-3">
          {violations.map((v) => (
            <div
              key={v.id}
              className="bg-[#141414] border border-gray-800 rounded-xl p-5 flex items-center justify-between gap-4 flex-wrap"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-2 flex-wrap">
                  <span className={`text-xs font-semibold px-2 py-1 rounded-md border ${TYPE_COLOR[v.violation_type] || 'text-gray-400 bg-gray-900 border-gray-700'}`}>
                    {TYPE_LABEL[v.violation_type] || v.violation_type}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded-md font-medium ${v.status === 'PAID' ? 'text-green-400 bg-green-950 border border-green-800' : 'text-red-400 bg-red-950 border border-red-800'}`}>
                    {v.status}
                  </span>
                </div>
                <p className="text-gray-400 text-sm">{formatDate(v.created_at)}</p>
                <p className="text-yellow-400 font-bold mt-1">₹{v.fine_amount}</p>
              </div>

              <PaymentButton
                violation={v}
                onSuccess={handlePaymentSuccess}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
