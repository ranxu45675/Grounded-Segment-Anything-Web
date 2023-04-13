import {createRouter, createWebHistory} from "vue-router";
import Main from "../components/Main.vue";
import About from '../components/About.vue';
const routes = [
    {
        path: '/main',
        name: 'main',
        component: Main
    },{
        path: '/about',
        name:'about',
        component: About
    }

]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router