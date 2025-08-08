<template>
    <div>
        <h2 class="text-xl font-semibold mb-4">Model Structure Viewer</h2>
        <div class="dropdown">
            <label for="model-select">Choose a model:</label>
            <select id="model-select" v-model="selectedModel" @change="loadModel">
                <option disabled value="">-- Select a model --</option>
                <option v-for="model in modelList" :key="model" :value="model">{{ model }}</option>
            </select>
        </div>
        <div v-if="modelStructure" class="graph-container">
            <div class="graph-nodes">
                <div v-for="(layer, name, index) in modelStructure" :key="name" class="node-item">
                    <div 
                        class="node-block"
                        @click="selectedNode = { name, ...layer }"
                        :class="{ 'selected': selectedNode && selectedNode.name === name }"
                        :style="{ width: getWidth(layer) }"
                    >
                        <div class="node-name">{{ name }}</div>
                        <div class="node-type">{{ layer.module_type }}</div>
                    </div>
                    <div v-if="index < Object.keys(modelStructure).length - 1" class="connector"></div>
                </div>
            </div>
            <div v-if="selectedNode" class="node-details">
                <h3>{{ selectedNode.name }}</h3>
                <div v-if="selectedNode.args && selectedNode.args.length > 0">
                    <h4>Arguments (args)</h4>
                    <div class="code-block">{{ formatArgs(selectedNode.args) }}</div>
                </div>
                <div v-if="selectedNode.kwargs && Object.keys(selectedNode.kwargs).length > 0">
                    <h4>Keyword Arguments (kwargs)</h4>
                    <div class="code-block">{{ formatKwargs(selectedNode.kwargs) }}</div>
                </div>
            </div>
            <div v-else class="node-details-placeholder">
                <p>Select a node to see its details.</p>
            </div>
        </div>
    </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import axios from 'axios';
import { useModelStore } from '../stores/modelStore';

const modelList = ref([]);
const modelStore = useModelStore();
const selectedModel = ref(modelStore.selectedModel);
const modelStructure = ref(null);
const selectedNode = ref(null);


function formatKwargs(kwargs) {
    if (!kwargs || Object.keys(kwargs).length === 0) {
        return "";
    }
    return Object.entries(kwargs)
        .map(([key, value]) => {
        if (typeof value === 'string') {
            return `${key}='${value}'`;
        }
        if (typeof value === 'object' && value !== null) {
            return `${key}=${JSON.stringify(value)}`;
        }
        return `${key}=${value}`;
        })
    .join(", ");
}

function formatArgs(args) {
    if (!args || args.length === 0) {
        return "";
    }
    return args.map(arg => {
        if (typeof arg === 'string') {
            return `'${arg}'`;
        }
        if (typeof arg === 'object' && arg !== null) {
            return JSON.stringify(arg);
        }
        return String(arg);
    }).join(", ");
}

function getHiddenDim(layer) {
    if (layer && layer.kwargs) {
        const dim = layer.kwargs.hidden_dim || layer.kwargs.hidden_size || layer.kwargs.d_model || layer.kwargs.n_embd || layer.kwargs.hidden_features || layer.kwargs.out_features || layer.kwargs.in_features;
        if (dim) return dim;
    }
    if (layer && layer.args) {
        // find the largest numeric argument, assuming it's the hidden dim
        const numericArgs = layer.args.filter(arg => typeof arg === 'number');
        if (numericArgs.length > 0) {
            return Math.max(...numericArgs);
        }
    }
    return null;
}

function getWidth(layer) {
    const hiddenDim = getHiddenDim(layer);
    if (hiddenDim) {
        const width = 180 + Math.min(120, hiddenDim / 10);
        return `${width}px`;
    }
    return '220px';
}

onMounted(async () => {
    try {
        const res = await axios.get('/api/models');
        if (Array.isArray(res.data)) {
            modelList.value = res.data;
        } else {
            console.error("Expected JSON array, got: ", res.data);
        }
    } catch (error) {
        console.error("Failed to fetch model list:", error);
    }
});

async function loadModel() {
    if (!selectedModel.value) return;
    selectedNode.value = null;
    try {
        modelStore.setModel(selectedModel.value);
        const response = await axios.get('/api/model-structure', {
            params: { model_name: selectedModel.value }
        });
        modelStructure.value = response.data;
        // Select the first node by default
        if (modelStructure.value && Object.keys(modelStructure.value).length > 0) {
            const firstName = Object.keys(modelStructure.value)[0];
            selectedNode.value = { name: firstName, ...modelStructure.value[firstName] };
        }
    } catch (error) {
        console.error("Failed to load model structure:", error);
    }
}
</script>

<style scoped>
.graph-container {
    display: flex;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin-top: 20px;
}
.graph-nodes {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    background: #fafafa;
    border-radius: 8px;
}
.node-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.node-block {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px 20px;
    margin-bottom: 10px;
    cursor: pointer;
    background-color: #ffffff;
    width: 220px;
    text-align: center;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.node-block:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.node-block.selected {
    border-color: #007bff;
    background-color: #e7f3ff;
    box-shadow: 0 0 12px rgba(0,123,255,0.5);
}
.node-name {
    font-weight: 600;
    font-size: 1.1em;
    margin-bottom: 5px;
    color: #333;
}
.node-type {
    font-style: italic;
    color: #555;
    font-size: 0.9em;
}
.connector {
    width: 2px;
    height: 30px;
    background-color: #d0d0d0;
    margin-bottom: 10px;
}
.node-details {
    padding: 0 20px;
    border-left: 1px solid #eee;
    flex-grow: 1;
    margin-left: 20px;
}
.node-details-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-grow: 1;
    color: #888;
    margin-left: 20px;
}
.node-details h3 {
    font-size: 1.5em;
    margin-top: 0;
    color: #000;
}
.node-details h4 {
    margin-top: 20px;
    margin-bottom: 10px;
    color: #333;
    border-bottom: 1px solid #eee;
    padding-bottom: 5px;
}
.code-block {
    background-color: #f7f7f7;
    padding: 15px;
    border-radius: 5px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    font-size: 0.9em;
    border: 1px solid #e0e0e0;
}
.dropdown {
    margin-bottom: 20px;
}
</style>

<!-- <template>
  <div class="graph-container">
    <svg :width="1200" :height="svgHeight">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#888" />
        </marker>
      </defs>
      <line
        v-for="edge in graph.edges"
        :key="`${edge.from}->${edge.to}`"
        :x1="getNode(edge.from)?.x + nodeWidth / 2"
        :y1="getNode(edge.from)?.y + nodeHeight / 2"
        :x2="getNode(edge.to)?.x + nodeWidth / 2"
        :y2="getNode(edge.to)?.y + nodeHeight / 2"
        stroke="#aaa"
        stroke-width="2"
        marker-end="url(#arrow)"
      />
      <g
        v-for="node in graph.nodes"
        :key="node.id"
        :transform="`translate(${node.x}, ${node.y})`"
        class="node-group"
        @click="selectNode(node)"
      >
        <rect
          :width="nodeWidth"
          :height="nodeHeight"
          rx="10"
          ry="10"
          :fill="getNodeColor(node)"
          stroke="#333"
          stroke-width="1"
        />
        <text x="10" y="24" font-size="14" fill="#fff" font-family="sans-serif">{{ node.name }}</text>
        <text x="10" y="44" font-size="12" fill="#ddd" font-family="monospace">{{ node.op }}</text>
      </g>
    </svg>

    <div v-if="selectedNode" class="node-details">
      <h3>{{ selectedNode.name }}</h3>
      <p><strong>op:</strong> {{ selectedNode.op }}</p>
      <p><strong>target:</strong> {{ selectedNode.target }}</p>
      <p><strong>group:</strong> {{ selectedNode.group_id }}</p>
      <p><strong>inputs:</strong> {{ selectedNode.inputs?.join(', ') }}</p>
      <p><strong>outputs:</strong> {{ selectedNode.outputs?.join(', ') }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import axios from 'axios';

const selectedModel = ref('');
const modelList = ref([]);
const graph = ref({ nodes: [], edges: [] });
const selectedNode = ref(null);
const nodeWidth = 160;
const nodeHeight = 60;

function getNode(id) {
  return graph.value.nodes.find(n => n.id === id);
}

function getNodeColor(node) {
  if (node.op === 'placeholder') return '#4A90E2';
  if (node.op === 'output') return '#D0021B';
  if (node.op === 'call_module') return '#7ED321';
  if (node.op === 'call_function') return '#9B9B9B';
  return '#888';
}

function selectNode(node) {
  selectedNode.value = node;
}

const svgHeight = computed(() => {
  const ys = graph.value?.nodes?.map(n => n.y ?? 0);
  return ys.length ? Math.max(...ys) + 200 : 500;
});

function layoutGraph(data) {
  const groupMap = new Map();
  const groupOrder = [];
  const yStep = 150;
  const xStep = 180;

  for (const node of data.nodes) {
    const group = node.group_id || 'ungrouped';
    console.log("Group map:");
    for (const [gid, nodes] of groupMap.entries()) {
        console.log(gid, nodes.map(n => n.name));
    }
    if (!groupMap.has(group)) {
      groupMap.set(group, []);
      groupOrder.push(group);
    }
    groupMap.get(group).push(node);
  }

  groupOrder.forEach((groupId, groupIndex) => {
    const nodes = groupMap.get(groupId);
    const y = groupIndex * yStep + 50;
    const totalWidth = nodes.length * xStep;
    const baseX = 600 - (totalWidth - xStep) / 2;

    nodes.forEach((node, i) => {
      node.x = baseX + i * xStep;
      node.y = y;
    });
  });

  return data;
}

async function loadModel() {
  if (!selectedModel.value) return;
  const res = await axios.get('/api/model-structure', {
    params: { model_name: selectedModel.value }
  });

  if (!res.data || !Array.isArray(res.data.nodes)) {
    console.error("Invalid model structure:", res.data);
    return;
  }

  graph.value = layoutGraph(res.data);
  selectedNode.value = null;
}

onMounted(async () => {
  const res = await axios.get('/api/models');
  modelList.value = res.data;

  // Auto-select and load first model
  if (modelList.value.length > 0) {
    selectedModel.value = modelList.value[1];
    await loadModel();
  }
});
</script>

<style scoped>
.graph-container {
  display: flex;
  flex-direction: row;
  gap: 20px;
  margin-top: 20px;
}
.node-details {
  padding: 10px;
  border-left: 1px solid #ccc;
}
.node-group:hover {
  cursor: pointer;
  opacity: 0.85;
}
</style> -->
