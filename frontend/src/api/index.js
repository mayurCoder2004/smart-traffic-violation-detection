import axios from 'axios'

const api = axios.create({
  baseURL: '',          // Vite proxy forwards to http://localhost:9000
  headers: { 'Content-Type': 'application/json' },
})

// Users
export const loginUser   = (plate) => api.post('/users/login', { license_plate: plate })
export const createUser  = (data)  => api.post('/users', data)
export const listUsers   = ()      => api.get('/users')

// Violations
export const getUserViolations = (userId) => api.get(`/violations/user/${userId}`)
export const getAllViolations   = ()       => api.get('/violations')
export const createViolation   = (data)   => api.post('/violations', data)

// Payments
export const createOrder  = (violationId) => api.post('/payments/create-order', { violation_id: violationId })
export const verifyPayment = (data)        => api.post('/payments/verify', data)
