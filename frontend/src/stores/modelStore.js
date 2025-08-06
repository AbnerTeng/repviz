import { defineStore } from 'pinia';

export const useModelStore = defineStore('modelStore', {
    state: () => ({
        selectedModel: '',
    }),
    actions: {
        setModel(modelName) {
            this.selectedModel = modelName;
        },
    },
});