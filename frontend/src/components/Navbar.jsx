import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()

  const link = (to, label) => (
    <Link
      to={to}
      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
        pathname === to
          ? 'bg-yellow-400 text-gray-900'
          : 'text-gray-400 hover:text-yellow-400'
      }`}
    >
      {label}
    </Link>
  )

  return (
    <nav className="border-b border-gray-800 bg-[#111] px-6 py-3 flex items-center justify-between">
      <span className="text-yellow-400 font-bold tracking-wide text-lg">
        TrafficGuard
      </span>
      <div className="flex gap-2">
        {link('/login',     'User Login')}
        {link('/dashboard', 'My Challans')}
        {link('/police',    'Police Dashboard')}
      </div>
    </nav>
  )
}
