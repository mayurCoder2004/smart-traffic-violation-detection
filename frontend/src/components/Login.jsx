import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { loginUser } from '../api'

export default function Login() {
  const [plate, setPlate]     = useState('')
  const [error, setError]     = useState('')
  const [loading, setLoading] = useState(false)
  const navigate              = useNavigate()

  const handleLogin = async (e) => {
    e.preventDefault()
    if (!plate.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await loginUser(plate.trim().toUpperCase())
      sessionStorage.setItem('user', JSON.stringify(res.data))
      navigate('/dashboard')
    } catch (err) {
      setError(err.response?.data?.error || 'Login failed. Check your license plate.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center min-h-[80vh] px-4">
      <div className="w-full max-w-md bg-[#141414] border border-gray-800 rounded-2xl p-8">
        <h1 className="text-2xl font-bold text-yellow-400 mb-2">Driver Login</h1>
        <p className="text-gray-500 text-sm mb-8">
          Enter your vehicle number plate to view your challans.
        </p>

        <form onSubmit={handleLogin} className="space-y-5">
          <div>
            <label className="block text-sm text-gray-400 mb-1">License Plate</label>
            <input
              type="text"
              value={plate}
              onChange={(e) => setPlate(e.target.value.toUpperCase())}
              placeholder="e.g. DL01AB1234"
              className="w-full bg-[#1e1e1e] border border-gray-700 rounded-lg px-4 py-3
                         text-white placeholder-gray-600 focus:outline-none focus:border-yellow-400
                         font-mono tracking-widest uppercase"
            />
          </div>

          {error && (
            <p className="text-red-400 text-sm bg-red-950 border border-red-800 rounded-lg px-4 py-3">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading || !plate.trim()}
            className="w-full bg-yellow-400 text-gray-900 font-bold py-3 rounded-lg
                       hover:bg-yellow-300 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            {loading ? 'Looking up...' : 'View My Challans'}
          </button>
        </form>

        <div className="mt-8 pt-6 border-t border-gray-800">
          <p className="text-gray-600 text-xs text-center">
            Police officer? Use the{' '}
            <a href="/police" className="text-yellow-400 hover:underline">
              Police Dashboard
            </a>{' '}
            instead.
          </p>
        </div>
      </div>
    </div>
  )
}
