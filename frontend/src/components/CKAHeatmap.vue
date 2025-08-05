<template>
  <div>
    <canvas ref="chart"></canvas>
  </div>
</template>

<script>
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

export default {
  props: ['chartData'],
  mounted() {
    this.renderChart();
  },
  methods: {
    renderChart() {
      const ctx = this.$refs.chart.getContext('2d');
      new Chart(ctx, {
        type: 'heatmap',
        data: {
          labels: this.chartData.model2_layers,
          datasets: [
            {
              label: 'CKA Similarity',
              data: this.chartData.cka_matrix,
              backgroundColor: (context) => {
                const value = context.dataset.data[context.dataIndex];
                const alpha = (value - 0.5) * 2; // Scale to 0-1
                return `rgba(75, 192, 192, ${alpha})`;
              },
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 1,
            },
          ],
        },
        options: {
          scales: {
            x: {
              type: 'category',
              labels: this.chartData.model1_layers,
            },
            y: {
              type: 'category',
              labels: this.chartData.model2_layers,
            },
          },
        },
      });
    },
  },
  watch: {
    chartData() {
      this.renderChart();
    }
  }
};
</script>
