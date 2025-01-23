// src/views/demos/SQLAgentView.vue
<template>
  <div class="demo-container">
    <DemoHeader
      :title="'SQL Agent Demo'"
      :credits="remainingCredits"
      @play-tutorial="playTutorial"
    />
    
    <div class="demo-content">
      <div class="comparison-view">
        <div class="before">
          <h3>Before Fine-tuning</h3>
          <CodeEditor v-model="query" @run="runOriginalModel" />
          <ResultView :results="originalResults" />
        </div>
        
        <div class="after">
          <h3>After Fine-tuning</h3>
          <CodeEditor v-model="query" @run="runFineTunedModel" />
          <ResultView :results="fineTunedResults" />
        </div>
      </div>
    </div>
    
    <VideoTutorial
      v-if="showTutorial"
      :videoUrl="tutorialUrl"
      @close="closeTutorial"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAuthStore } from '@/stores/auth'
import DemoHeader from '@/components/DemoHeader.vue'
import CodeEditor from '@/components/CodeEditor.vue'
import ResultView from '@/components/ResultView.vue'
import VideoTutorial from '@/components/VideoTutorial.vue'
import type { DemoResult } from '@/types'

const authStore = useAuthStore()
const remainingCredits = computed(() => authStore.remainingCredits('sql-agent'))

const query = ref<string>('')
const originalResults = ref<DemoResult | null>(null)
const fineTunedResults = ref<DemoResult | null>(null)
const showTutorial = ref<boolean>(false)
const tutorialUrl = 'path/to/tutorial.mp4'

const runOriginalModel = async (): Promise<void> => {
  try {
    await authStore.useDemo('sql-agent')
    // API call to original model
  } catch (error) {
    // Handle error
  }
}

const runFineTunedModel = async (): Promise<void> => {
  try {
    await authStore.useDemo('sql-agent')
    // API call to fine-tuned model
  } catch (error) {
    // Handle error
  }
}

const playTutorial = (): void => {
  showTutorial.value = true
}

const closeTutorial = (): void => {
  showTutorial.value = false
}
</script>