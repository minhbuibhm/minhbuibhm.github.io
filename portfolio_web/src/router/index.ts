// src/router/index.ts
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/HomeView.vue')
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/LoginView.vue')
  },
  {
    path: '/demos',
    name: 'Demos',
    component: () => import('@/views/DemosView.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: 'sql-agent',
        name: 'SQLAgent',
        component: () => import('@/views/demos/SQLAgentView.vue')
      },
      {
        path: 'function-calling',
        name: 'FunctionCalling',
        component: () => import('@/views/demos/FunctionCallingView.vue')
      },
      {
        path: 'rag-chatbot',
        name: 'RAGChatbot',
        component: () => import('@/views/demos/RAGChatbotView.vue')
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const authStore = useAuthStore()
  
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
  } else {
    next()
  }
})

export default router