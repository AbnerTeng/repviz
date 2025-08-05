<template>
  <div id="app">
    <ThemeSwitcher />
    <h1>Neural Layer Analysis</h1>
    <ModelGraph />
    <ActivationScatter :activations="activations" />
    <!-- <WeightHistorgram :weights="weights" /> -->
  </div>
</template>

<script>
import ThemeSwitcher from './components/ThemeSwitcher.vue';
import ModelGraph from './components/ModelGraph.vue';
// import ActivationScatter from './components/ActivationScatter.vue';
// import WeightHistorgram from './components/WeightHistorgram.vue';

export default {
  name: 'App',
  components: {
    ModelGraph,
    // ActivationScatter,
    // WeightHistorgram,
    ThemeSwitcher,
  },
  data() {
    return {
      // activations: activations,
      // weights: weights,
    };
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  
  margin-top: 60px;
}
</style>
