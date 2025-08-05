<template>
    <div>
        <label for="layer-select">Select Output Layer:</label>
        <select v-model="selectedLayer" id="layer-select">
            <option v-for="layer in layerList" :key="layer" :value="layer">
                {{ layer }}
            </option>
        </select>
    </div>
</template>

<script>
import Plotly from "plotly.js-dist-min";
export default {
    name: "WeightHistogram",
    props: {
        weights: Object,
    },
    data() {
        return {
            selectedLayer: null,
        };
    },
    computed: {
        layerList() {
            return Object.keys(this.weights);
        },
    },
    watch: {
        selectedLayer: "drawWeightHistogram",
    },
    mounted() {
        fetch("http://localhost:8000/api/weights")
            .then(response => response.json())
            .then(data => {
                this.weights = data;
                if (this.layerList.length > 0) {
                    this.selectedLayer = this.layerList[0];
                    this.drawWeightHistogram();
                }
            });
    },
    methods: {
        drawHistogram() {
            if (!this.selectedLayer) return;

            // Flatten 2D weight array into 1D
            const weightMatrix = this.weights[this.selectedLayer];
            const flattened = weightMatrix.flat();

            const trace = {
                x: flattened,
                type: "histogram",
                marker: {
                    color: "skyblue",
                    line: {
                        color: "black",
                        width: 1,
                    },
                },
            };
            const layout = {
                title: `Weight Distribution: ${this.selectedLayer}`,
                xaxis: { title: "Weight Value" },
                yaxis: { title: "Count" },
                bargap: 0.05,
            };
            Plotly.newPlot("weight-histogram", [trace], layout, { responsive: true });
        },
    },
};
</script>

<style scoped>
#weight-histogram {
    width: 100%;
    height: 400px;
}
</style>
