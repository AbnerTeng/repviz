<template>
  <div>
    <label for="layer-select">Select Output Layer:</label>
    <select v-model="selectedLayer" id="layer-select">
      <option v-for="(layer, i) in layerList.slice(1)" :key="layer" :value="layer">
        {{ layer }}
      </option>
    </select>

    <div id="scatter-plot"></div>
  </div>
</template>

<script>
import Plotly from "plotly.js-dist-min";

export default {
  name: "ActivationScatter",
  props: {
    activations: Object,
  },
  data() {
    return {
      selectedLayer: null,
    };
  },
  computed: {
    layerList() {
      return Object.keys(this.activations);
    },
  },
  watch: {
    selectedLayer: "drawScatterPlot",
  },
  mounted() {
    fetch("http://localhost:8000/api/activations")
      .then(response => response.json())
      .then(data => {
        this.activations = data;
        if (this.layerList.length > 1) {
          this.selectedLayer = this.layerList[1];
          this.drawScatterPlot();
        }
      });
  },
  methods: {
    drawScatterPlot() {
      const currentIndex = this.layerList.indexOf(this.selectedLayer);
      if (currentIndex <= 0) return;
  
      const prevLayer = this.layerList[currentIndex - 1];
      const x = this.activations[prevLayer][0];
      const y = this.activations[this.selectedLayer][0]; // same as x if input = output; else modify

      const trace = {
        x,
        y,
        mode: "markers",
        type: "scatter",
        marker: {
          color: y,
          colorscale: "Viridis",
          size: 6,
          line: {
            width: 0.5,
            color: "black",
          },
        },
      };
      const layout = {
        title: `Activations of Layer: ${this.selectedLayer}`,
        xaxis: { title: "Input" },
        yaxis: { title: "Activation" },
      };
      Plotly.newPlot("scatter-plot", [trace], layout, { responsive: true });
    },
  },
};
</script>
