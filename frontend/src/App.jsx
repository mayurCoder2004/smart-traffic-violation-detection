import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Login from './components/Login'
import PoliceDashboard from './components/PoliceDashboard'
import TrafficSignal from './components/TrafficSignal'
import ChallanPolice from './components/ChallanPolice'
import ChallanUser   from './components/ChallanUser'

export default function App() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-grid text-gray-100">
      <Navbar />
      {/* AnimatePresence enables exit animations when routes change */}
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/"            element={<Navigate to="/login" replace />} />
          <Route path="/login"       element={<Login />} />
          <Route path="/dashboard"   element={<ChallanUser />} />
          <Route path="/police"      element={<PoliceDashboard />} />
          <Route path="/signal"      element={<TrafficSignal />} />
          <Route path="/scanner"     element={<ChallanPolice />} />
          <Route path="/my-challans" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </AnimatePresence>
    </div>
  )
}
