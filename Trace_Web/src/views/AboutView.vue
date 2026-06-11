<template>
  <section class="about-page">
    <article class="panel">
      <p class="eyebrow">Method Overview</p>
      <h1>TracePredict 的展示逻辑</h1>
      <p class="lead">
        这个前端页主要用于把“历史轨迹、预测轨迹、真实后续轨迹”放在同一张图里，让模型结果更容易解释。
      </p>
    </article>

    <article class="panel">
      <h2>页面包含什么</h2>
      <ul>
        <li>历史观测轨迹：表示用户已经发生的移动片段。</li>
        <li>预测轨迹：表示模型根据 Markov 转移和偏好信息推断出的后续位置。</li>
        <li>真实轨迹：表示实际后续点位，用于和预测结果对比。</li>
        <li>ADE / FDE：用于直观看预测偏差的两个简单指标。</li>
      </ul>
    </article>

    <article class="panel">
      <h2>后续怎么接真数据</h2>
      <ul>
        <li>把 `src/data/traceSamples.ts` 替换成后端接口返回的数据。</li>
        <li>保持 `observed / actualFuture / predictedFuture` 三段结构不变。</li>
        <li>如果后面要接地图 SDK，可以把 SVG 组件替换成地图图层渲染。</li>
      </ul>
    </article>
  </section>
</template>

<style scoped>
.about-page {
  display: grid;
  gap: 1rem;
}

.panel {
  background: #fff;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 24px;
  padding: 1.5rem;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
}

.eyebrow {
  color: #2563eb;
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

h1,
h2 {
  color: #0f172a;
  font-weight: 700;
}

h1 {
  margin-top: 0.45rem;
  font-size: 2rem;
}

h2 {
  font-size: 1.2rem;
}

.lead {
  margin-top: 0.8rem;
  color: #475569;
}

ul {
  margin-top: 0.75rem;
  padding-left: 1.1rem;
  color: #475569;
}

li + li {
  margin-top: 0.45rem;
}
</style>
