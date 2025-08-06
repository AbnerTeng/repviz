<template>
  <div>
    <h2 class="text-xl font-semibold mb-4">Weight Plots</h2>

    <div v-if="selectedModel && layerList.length > 1" class="dropdown">
      <label for="layer-select">Choose a layer:</label>
      <select id="layer-select" v-model="selectedLayer">
        <option disabled value="">-- Select a layer --</option>
        <option v-for="layer in layerList" :key="layer" :value="layer">{{ layer }}</option>
      </select>
    </div>
    <div class="plot-row">
      <div v-if="selectedLayer" ref="plotContainerHist" class="weight-plot-container"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
import { useModelStore } from '../stores/modelStore';
import axios from 'axios';
import Plotly from 'plotly.js-dist-min';

const plotContainerHist = ref(null);
const activations = ref({});
const selectedLayer = ref('');
const modelStore = useModelStore();
const selectedModel = computed(() => modelStore.selectedModel);
const layerList = computed(() => Object.keys(activations.value));

watch(selectedModel, async (model) => {
  if (!model) return;
  selectedLayer.value = '';
  try {
    const res = await axios.get('/api/weights', {
      params: { model_name: model },
    });
    activations.value = res.data;
    if (layerList.value.length > 1) {
      selectedLayer.value = layerList.value[1]; // auto select
    }
  } catch (err) {
    console.error("Failed to load activations:", err);
  }
});
watch(selectedLayer, async (layer) => {
  if (!selectedModel.value || !layer) return;

  try {
    const res = await axios.get('/api/weights/plots', {
      params: {
        model_name: selectedModel.value,
        layer_y: layer,
      },
    });

    const { weights, layer_y } = res.data;

    Plotly.newPlot(plotContainerHist.value, [{
      x: weights,
      type: 'histogram',
      nbinsx: 50,
      name: `${layer_y} weight distribution`,
      marker: {
        color: 'blue',
        line: {
          color: 'black',
          width: 1,
        }
      },
      opacity: 0.6,
      histnorm: '',
    }], {
      title: {
        text: `Weight Distribution for ${layer_y}`,
        font: { size: 16 },
      },
      xaxis: {
        title: {
          text: `Weight values of ${layer_y}`,
          font: { size: 14 },
        },
      },
      yaxis: {
        title: {
          text: 'Frequency',
          font: { size: 14 },
        },
      },
      margin: { t: 60, l: 60, r: 30, b: 60 },
      plot_bgcolor: 'white',
    }, { responsive: true });
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
.weight-plot-container {
  flex: 1;
  min-width: 0;
}
.dropdown {
  margin-bottom: 20px;
}
</style>
