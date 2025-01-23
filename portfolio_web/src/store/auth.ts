// src/stores/auth.ts
import { defineStore } from 'pinia'
import axios from 'axios'
import { AuthState, LoginCredentials, User, DemoCredits } from '@/types'

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    user: null,
    token: localStorage.getItem('token'),
    demoCredits: {}
  }),
  
  getters: {
    isAuthenticated: (state: AuthState): boolean => !!state.token,
    remainingCredits: (state: AuthState) => (demoId: string): number => 
      state.demoCredits[demoId] || 0
  },
  
  actions: {
    async login(credentials: LoginCredentials): Promise<void> {
      try {
        const response = await axios.post<{
          token: string;
          user: User;
          demoCredits: DemoCredits;
        }>('/api/auth/login', credentials)
        
        this.token = response.data.token
        this.user = response.data.user
        this.demoCredits = response.data.demoCredits
        localStorage.setItem('token', this.token)
        axios.defaults.headers.common['Authorization'] = `Bearer ${this.token}`
      } catch (error) {
        throw new Error('Login failed')
      }
    },
    
    async useDemo(demoId: string): Promise<void> {
      if (this.demoCredits[demoId] > 0) {
        this.demoCredits[demoId]--
        await axios.post(`/api/demos/${demoId}/use`)
      } else {
        throw new Error('No demo credits remaining')
      }
    }
  }
})
