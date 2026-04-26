import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Login from './components/Login'
import UserDashboard from './components/UserDashboard'
import PoliceDashboard from './components/PoliceDashboard'

export default function App() {
  return (
    <div className="min-h-screen bg-[#0d0d0d] text-gray-100">
      <Navbar />
      <Routes>
        <Route path="/"        element={<Navigate to="/login" replace />} />
        <Route path="/login"   element={<Login />} />
        <Route path="/dashboard" element={<UserDashboard />} />
        <Route path="/police"  element={<PoliceDashboard />} />
      </Routes>
    </div>
  )
}
