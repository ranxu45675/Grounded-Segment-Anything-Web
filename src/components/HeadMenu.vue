<template>
  <el-menu
    :default-active="activeIndex"
    class="el-menu-demo"
    mode="horizontal"
    :ellipsis="false"
  >
    <el-menu-item index="0">
      <router-link :to="{name:'main'}">主界面</router-link>
    </el-menu-item>
        <el-menu-item index="1">
      <router-link :to="{name:'about'}">关于我</router-link>
    </el-menu-item>
  </el-menu>
</template>


<script lang="ts" setup>
import { ref } from 'vue'

const activeIndex = ref('1')
</script>

<style scoped>
  * {
    text-decoration: none;
  }
</style>