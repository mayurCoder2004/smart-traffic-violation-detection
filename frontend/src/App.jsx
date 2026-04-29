import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Login from './components/Login'
import UserDashboard from './components/UserDashboard'
import PoliceDashboard from './components/PoliceDashboard'
import TrafficSignal from './components/TrafficSignal'

export default function App() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-grid text-gray-100">
      <Navbar />
      {/* AnimatePresence enables exit animations when routes change */}
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/"          element={<Navigate to="/login" replace />} />
          <Route path="/login"     element={<Login />} />
          <Route path="/dashboard" element={<UserDashboard />} />
          <Route path="/police"    element={<PoliceDashboard />} />
          <Route path="/signal"    element={<TrafficSignal />} />
        </Routes>
      </AnimatePresence>
    </div>
  )
}
