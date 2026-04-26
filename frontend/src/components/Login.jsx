import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, useMotionValue, useTransform } from 'framer-motion'
import { Car, AlertCircle, ArrowRight, Loader2, ShieldCheck } from 'lucide-react'
import { loginUser } from '../api'

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

export default function Login() {
  /* ── All existing state/logic unchanged ── */
  const [plate, setPlate]     = useState('')
  const [error, setError]     = useState('')
  const [loading, setLoading] = useState(false)
  const navigate              = useNavigate()

  /* 3D tilt — only active on pointer devices */
  const cardRef = useRef(null)
  const mouseX  = useMotionValue(0)
  const mouseY  = useMotionValue(0)
  const rotateX = useTransform(mouseY, [-0.5, 0.5], [6, -6])
  const rotateY = useTransform(mouseX, [-0.5, 0.5], [-6, 6])

  const onMouseMove = (e) => {
    const rect = cardRef.current?.getBoundingClientRect()
    if (!rect) return
    mouseX.set((e.clientX - rect.left) / rect.width  - 0.5)
    mouseY.set((e.clientY - rect.top)  / rect.height - 0.5)
  }
  const onMouseLeave = () => { mouseX.set(0); mouseY.set(0) }

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
    <PageWrapper>
      {/* Full-viewport centering; min-h accounts for navbar (~53px) */}
      <div className="relative flex items-center justify-center
                      min-h-[calc(100dvh-53px)] px-3 sm:px-6 py-8 overflow-hidden">

        {/* Ambient glow blobs */}
        <div className="absolute top-1/4 left-1/5 w-64 sm:w-80 h-64 sm:h-80
                        bg-yellow-400/5 rounded-full blur-[90px] pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-44 sm:w-56 h-44 sm:h-56
                        bg-yellow-400/4 rounded-full blur-[65px] pointer-events-none" />

        {/* 3D tilt wrapper */}
        <motion.div
          ref={cardRef}
          style={{ rotateX, rotateY, transformPerspective: 1100 }}
          onMouseMove={onMouseMove}
          onMouseLeave={onMouseLeave}
          className="w-full max-w-sm sm:max-w-md"
        >
          {/* Glassmorphism card */}
          <div className="glass rounded-2xl p-5 sm:p-8
                          shadow-[0_20px_56px_rgba(0,0,0,0.65)] relative overflow-hidden">

            {/* Inner-top highlight */}
            <div className="absolute top-0 left-0 right-0 h-px
                            bg-gradient-to-r from-transparent via-white/10 to-transparent" />

            {/* Header */}
            <div className="flex items-center gap-3 mb-5 sm:mb-6">
              <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl bg-yellow-400/10 border border-yellow-400/30
                              flex items-center justify-center animate-pulse-ring flex-shrink-0">
                <Car className="w-5 h-5 text-yellow-400" />
              </div>
              <div>
                <h1 className="text-lg sm:text-xl font-bold text-white leading-tight">Driver Login</h1>
                <p className="text-gray-500 text-xs mt-0.5 flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3 text-yellow-400/60" />
                  VioLens · Violation Detection System
                </p>
              </div>
            </div>

            <p className="text-gray-500 text-sm mb-5 sm:mb-7 leading-relaxed">
              Enter your vehicle registration number to view your challans and outstanding fines.
            </p>

            <form onSubmit={handleLogin} className="space-y-4 sm:space-y-5">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-widest mb-2">
                  License Plate Number
                </label>
                <input
                  type="text"
                  value={plate}
                  onChange={(e) => setPlate(e.target.value.toUpperCase())}
                  placeholder="e.g. DL01AB1234"
                  /* Larger touch target on mobile */
                  className="w-full bg-white/[0.04] border border-white/[0.09] rounded-xl
                             px-4 py-3 sm:py-3.5
                             text-white placeholder-gray-600 font-mono tracking-[0.15em] sm:tracking-[0.18em]
                             uppercase text-base sm:text-lg
                             focus:outline-none focus:border-yellow-400/50
                             focus:shadow-[0_0_0_3px_rgba(240,192,64,0.12),0_0_20px_rgba(240,192,64,0.08)]
                             transition-all duration-200"
                />
              </div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.2 }}
                  className="flex items-start gap-2.5 text-red-400 text-sm
                             bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3"
                >
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{error}</span>
                </motion.div>
              )}

              <button
                type="submit"
                disabled={loading || !plate.trim()}
                className="relative w-full overflow-hidden font-bold py-3 sm:py-3.5 rounded-xl
                           bg-yellow-400 text-gray-900 text-sm
                           hover:bg-yellow-300 disabled:opacity-40 disabled:cursor-not-allowed
                           transition-all duration-200
                           hover:shadow-[0_0_28px_rgba(240,192,64,0.45)]
                           active:scale-[0.98] group"
              >
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300
                                bg-gradient-to-r from-transparent via-white/20 to-transparent
                                -translate-x-full group-hover:translate-x-full
                                transition-transform duration-700 pointer-events-none" />
                <span className="relative flex items-center justify-center gap-2">
                  {loading ? (
                    <><Loader2 className="w-4 h-4 animate-spin" />Looking up...</>
                  ) : (
                    <>View My Challans
                      <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-200" />
                    </>
                  )}
                </span>
              </button>
            </form>

            <div className="mt-6 sm:mt-7 pt-5 sm:pt-6 border-t border-white/[0.06]">
              <p className="text-gray-600 text-xs text-center">
                Police officer?{' '}
                <a href="/police" className="text-yellow-400 hover:text-yellow-300 font-semibold transition-colors">
                  Access Police Dashboard →
                </a>
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </PageWrapper>
  )
}
