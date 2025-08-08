<template>
  <div>
    <h2 class="text-xl font-semibold mb-4">CKA similarity</h2>

    <div v-if="selectedModel && modelList.length >= 0" class="dropdown">
      <label for="another-model-select">Choose a model to compare:</label>
      <select id="another-model-select" v-model="selectedCompareModel">
        <option disabled value="">-- Select another model --</option>
        <option v-for="model in modelList" :key="model" :value="model">{{ model }}</option>
      </select>
    </div>
    <div class="plot-row">
      <div v-if="selectedCompareModel" ref="plotCKA" class="heatmap-container"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import { useModelStore } from '../stores/modelStore';
import axios from 'axios';
import Plotly from 'plotly.js-dist-min';

const plotCKA = ref(null);
const selectedCompareModel = ref('');
const modelStore = useModelStore();
const selectedModel = computed(() => modelStore.selectedModel);
const modelList = ref([]);

onMounted(async () => {
  try {
    const res = await axios.get('/api/models');
    if (Array.isArray(res.data)) {
      modelList.value = res.data.filter(model => model !== selectedModel.value);
    } else {
      console.error("Expected model list to be an array:", res.data);
    }
  } catch (err) {
    console.error("Failed to fetch models:", err);
  }
});

console.log("modelList", modelList.value);


async function getLayers(model) {
  if (!model) return;
  try {
    const res = await axios.get('/api/activations', {
      params: { model_name: model },
    });
    return Object.keys(res.data);
  } catch (err) {
    console.error("Failed to load model structure:", err);
    return [];
  }
};


watch(selectedCompareModel, async (model) => {
  if (!selectedModel.value || !model) return;

  try {
    const res = await axios.get('/api/cka-similarity', {
      params: {
        model1: selectedModel.value,
        model2: model,
      },
    });
    const m1 = res.data.model1;
    const m2 = res.data.model2;
    const m1_layers = await getLayers(m1);
    const m2_layers = await getLayers(m2);
    const xLabels = Array.from(m2_layers);
    const yLabels = Array.from(m1_layers).reverse();
    const trace = {
      z: res.data.cka_similarity,
      x: xLabels,
      y: yLabels,
      type: 'heatmap',
      colorscale: 'Viridis',
      reversescale: false,
      showscale: true,
    };

    const layout = {
      title: 'CKA Similarity Heatmap',
      xaxis: {
        title: 'used_activs2',
        side: 'top',
        tickangle: -45,
      },
      yaxis: {
        title: 'used_activs1 (reversed)',
        autorange: 'reversed',  // to match seaborn’s y-axis order
      },
      margin: { t: 100 },
    };

    Plotly.newPlot(plotCKA.value, [trace], layout);
  } catch (err) {
    console.error("Failed to draw scatter plot:", err);
  }
});
</script>

<style scoped>
.plot-row {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  margin-top: 20px;
}
.heatmap-container {
  flex: 1;
  min-width: 0;
}
.dropdown {
  margin-bottom: 20px;
}
</style>
