<script setup>
import Test from './components/Test.vue'
import HeadMenu from "@/components/HeadMenu.vue";
</script>

<template>
  <el-container>
    <el-header>
      <HeadMenu></HeadMenu>
    </el-header>
    <el-main>
      <router-view></router-view>
    </el-main>
  </el-container>
</template>

<style scoped>

</style>
