// src/components/DemoHeader.vue
<template>
  <div class="demo-header">
    <h2>{{ title }}</h2>
    <div class="controls">
      <div class="credits">
        Credits remaining: {{ credits }}
      </div>
      <button @click="$emit('play-tutorial')" class="tutorial-btn">
        Watch Tutorial
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string;
  credits: number;
}

defineProps<Props>()
defineEmits<{
  (e: 'play-tutorial'): void;
}>()
</script>