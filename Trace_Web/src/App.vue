<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="brand-kicker">TracePredict</p>
        <h1 class="brand-title">轨迹预测演示前端</h1>
      </div>

      <nav class="nav">
        <RouterLink to="/">轨迹展示</RouterLink>
        <RouterLink to="/about">说明</RouterLink>
      </nav>
    </header>

    <RouterView />
  </div>
</template>

<style scoped>
.app-shell {
  display: grid;
  gap: 1.5rem;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.brand-kicker {
  color: #2563eb;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.82rem;
  font-weight: 700;
}

.brand-title {
  font-size: 1.6rem;
  font-weight: 700;
  color: #0f172a;
}

.nav {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.nav a {
  padding: 0.7rem 1rem;
  border-radius: 999px;
  color: #334155;
  background: rgba(148, 163, 184, 0.12);
}

.nav a.router-link-exact-active {
  color: #fff;
  background: #2563eb;
}

@media (max-width: 720px) {
  .topbar {
    align-items: start;
    flex-direction: column;
  }
}
</style>
