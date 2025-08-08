<template>
  <div id="app">
    <ThemeSwitcher />
    <h1>Neural Layer Analysis</h1>
    <ModelGraph />
    <ActivationScatter />
    <WeightHistogram />
    <GradientHistogram />
    <CKAHeatmap />
  </div>
</template>

<script setup>
import ThemeSwitcher from './components/ThemeSwitcher.vue';
import ModelGraph from './components/ModelGraph.vue';
import ActivationScatter from './components/ActivationScatter.vue';
import WeightHistogram from './components/WeightHistogram.vue';
import CKAHeatmap from './components/CKAHeatmap.vue';
import GradientHistogram from './components/GradientHistogram.vue';
</script>

<style scoped>
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  width: 100%;
  margin-top: 60px;
}
</style>
