import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, User, ClipboardList, BarChart2, Menu, X } from 'lucide-react'

const NAV_ITEMS = [
  { to: '/login',     label: 'User Login',      Icon: User },
  { to: '/dashboard', label: 'My Challans',      Icon: ClipboardList },
  { to: '/police',    label: 'Police Dashboard', Icon: BarChart2 },
]

export default function Navbar() {
  const { pathname }    = useLocation()
  const [open, setOpen] = useState(false)

  return (
    <>
      <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-[rgba(5,5,8,0.90)] backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 py-3 flex items-center justify-between gap-2">

          {/* Brand logo */}
          <Link to="/" className="flex items-center gap-2 group flex-shrink-0" onClick={() => setOpen(false)}>
            <div className="relative">
              <div className="w-8 h-8 rounded-lg bg-yellow-400/10 border border-yellow-400/30
                              flex items-center justify-center transition-all duration-300
                              group-hover:bg-yellow-400/20 group-hover:border-yellow-400/60
                              animate-glow-pulse">
                <Shield className="w-4 h-4 text-yellow-400" fill="currentColor" />
              </div>
              <div className="absolute inset-0 rounded-lg bg-yellow-400/20 blur-md
                              opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10" />
            </div>
            <span className="text-white font-bold tracking-tight text-base sm:text-[17px]">
              Vio<span className="text-yellow-400">Lens</span>
            </span>
          </Link>

          {/* Desktop nav — hidden on mobile */}
          <div className="hidden md:flex items-center gap-0.5">
            {NAV_ITEMS.map(({ to, label, Icon }) => {
              const active = pathname === to
              return (
                <Link
                  key={to}
                  to={to}
                  className={`relative flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium
                              transition-colors duration-150 ${
                    active ? 'text-gray-900' : 'text-gray-400 hover:text-white hover:bg-white/[0.05]'
                  }`}
                >
                  {active && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 rounded-lg bg-yellow-400"
                      initial={false}
                      transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                    />
                  )}
                  <Icon className="w-4 h-4 relative z-10 flex-shrink-0" />
                  <span className="relative z-10">{label}</span>
                </Link>
              )
            })}
          </div>

          {/* Mobile: icon-strip + hamburger */}
          <div className="flex items-center gap-1 md:hidden">
            {/* Icon-only strip (sm screens) */}
            <div className="flex items-center gap-0.5 sm:flex hidden">
              {NAV_ITEMS.map(({ to, Icon }) => {
                const active = pathname === to
                return (
                  <Link
                    key={to}
                    to={to}
                    className={`relative p-2.5 rounded-lg transition-colors duration-150 ${
                      active ? 'text-gray-900' : 'text-gray-400 hover:text-white hover:bg-white/[0.05]'
                    }`}
                  >
                    {active && (
                      <motion.div
                        layoutId="nav-pill-sm"
                        className="absolute inset-0 rounded-lg bg-yellow-400"
                        initial={false}
                        transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                      />
                    )}
                    <Icon className="w-4 h-4 relative z-10" />
                  </Link>
                )
              })}
            </div>

            {/* Hamburger button (xs screens only) */}
            <button
              onClick={() => setOpen((v) => !v)}
              className="sm:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/[0.05]
                         transition-colors duration-150"
              aria-label="Toggle menu"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile drawer — xs screens only */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="sm:hidden sticky top-[53px] z-40 border-b border-white/[0.06]
                       bg-[rgba(5,5,8,0.97)] backdrop-blur-xl overflow-hidden"
          >
            <div className="px-3 py-2 space-y-0.5">
              {NAV_ITEMS.map(({ to, label, Icon }) => {
                const active = pathname === to
                return (
                  <Link
                    key={to}
                    to={to}
                    onClick={() => setOpen(false)}
                    className={`flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium
                                transition-colors duration-150 ${
                      active
                        ? 'bg-yellow-400 text-gray-900'
                        : 'text-gray-400 hover:text-white hover:bg-white/[0.05]'
                    }`}
                  >
                    <Icon className="w-4 h-4 flex-shrink-0" />
                    {label}
                  </Link>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
