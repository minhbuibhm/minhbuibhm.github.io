// src/views/HomeView.vue
<template>
  <div class="home">
    <header class="hero">
      <h1>LLM Projects Portfolio</h1>
      <p>Exploring the capabilities of Large Language Models</p>
    </header>
    
    <section class="projects">
      <ProjectCard
        v-for="project in projects"
        :key="project.id"
        :project="project"
        @click="navigateToDemo(project.demoPath)"
      />
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import ProjectCard from '@/components/ProjectCard.vue'
import { Project } from '@/types'

const router = useRouter()

const projects = ref<Project[]>([
  {
    id: 1,
    title: 'SQL Agent with Fine-tuning',
    description: 'Compare the performance of SQL agents before and after fine-tuning',
    demoPath: '/demos/sql-agent'
  },
  {
    id: 2,
    title: 'LLM Function Calling',
    description: 'Demonstrate advanced function calling capabilities in LLMs',
    demoPath: '/demos/function-calling'
  },
  {
    id: 3,
    title: 'RAG Chatbot',
    description: 'Interactive chatbot with Retrieval Augmented Generation',
    demoPath: '/demos/rag-chatbot'
  }
])

const navigateToDemo = (path: string): void => {
  router.push(path)
}
</script>