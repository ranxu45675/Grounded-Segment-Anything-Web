<template >
  <el-card
      v-loading="loading"
      element-loading-text="Loading..."
      :element-loading-spinner="svg"
      element-loading-svg-view-box="-10, -10, 50, 50"
      element-loading-background="rgba(122, 122, 122, 0.8)"
  >
      <el-upload
    ref="upload"
    class="upload-demo"
    action="http://127.0.0.1:5000/detect"
    :limit="1"
    :on-exceed="handleExceed"
    :auto-upload="false"
    :on-success="getData"
    :data="upData"
    list-type="picture"
  >
    <template #trigger>
      <el-button type="primary">select file</el-button>
    </template>
    <el-input v-model="body" style="width: 180px;margin-left: 20px" placeholder="enter main body"></el-input>
    <el-input v-model="prompt" style="width: 320px;margin-left: 20px" placeholder="enter prompt"></el-input>
    <el-button class="ml-3" type="success" @click="submitUpload" style="margin-left: 20px">
      upload to server
    </el-button>
  </el-upload>
    <div class="demo-image__placeholder">
      <div v-for="(imageData, index) in images" :key="index">
        <el-image :src="imageData"/>
        <div style="float: right">
          {{step[index]}}
        </div>
      </div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import {reactive, ref} from 'vue'
import { genFileId } from 'element-plus'
import type { UploadInstance, UploadProps, UploadRawFile } from 'element-plus'

const upload = ref<UploadInstance>()
let images:any = reactive([])
let prompt:any = ref()
let body:any = ref()
let loading:any = ref(false)
let step = ['第一步:输入原图','第二步:处理图片','第三步:提前所需特征','第四部:获取所需要替换的位置','生成结果']

let upData = reactive<param>({
  prompt:null,
  body:null
})

class param {
  prompt:any
  body:any
}



const handleExceed: UploadProps['onExceed'] = (files) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
}

const submitUpload = () => {
  upData.prompt = prompt.value;
  upData.body = body.value;
  upload.value!.submit();
  loading = true
};

const getData = (response) => {
   loading = false
   images.length=0
   response.images.forEach(item=>{
   images.push('data:image/jpeg;base64,' + item)
 })
  console.log(images)
}

</script>


<style scoped>
.demo-image__placeholder .block {
  padding: 30px 0;
  text-align: center;
  border-right: solid 1px var(--el-border-color);
  display: inline-block;
  width: 49%;
  box-sizing: border-box;
  vertical-align: top;
}
.demo-image__placeholder .demonstration {
  display: block;
  color: var(--el-text-color-secondary);
  font-size: 14px;
  margin-bottom: 20px;
}
.demo-image__placeholder .el-image {
  padding: 0 5px;
  max-width: 500px;
  max-height: 300px;
}

.demo-image__placeholder .dot {
  animation: dot 2s infinite steps(3, start);
  overflow: hidden;
}
</style>