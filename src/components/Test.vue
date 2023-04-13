<template>
  <div>
    <input type="file" @change="onFileSelected">
    <button @click="uploadAndLoad">Upload and Load Images</button>
    <div v-for="(imageData, index) in images" :key="index">
      <img :src="imageData" style="width: 200px;height: 200px" alt=""/>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import axios from 'axios';

export default defineComponent({
  data() {
    return {
      currentImg: 1,
      selectedFile: null as File | null,
      images: [] as string[],
    };
  },
  methods: {
    onFileSelected(event: Event): void {
      this.selectedFile = (event.target as HTMLInputElement).files?.[0] || null;
    },
    async uploadAndLoad(): Promise<void> {
      if (this.selectedFile) {
        const formData = new FormData();
        formData.append('file', this.selectedFile);
        const response = await axios.post('http://127.0.0.1:5000/detect', formData);
        const imageDatas = response.data.images;
        this.images = imageDatas.map(imageData => 'data:image/jpeg;base64,' + imageData);
      }
    },
  },
});
</script>
