# TracePredict
<<<<<<< HEAD
Trajectory prediction using large language model
=======

基于轨迹数据、POI 信息、聚类和马尔科夫链的出行轨迹预测项目，仓库内同时包含一个用于可视化预测结果的前端 `Trace_Web`。

## 主要入口

- `main.py`: 当前 Python 主入口，支持 Markov 评估和导出前端样本
- `Markov_chain.py`: 转移矩阵与评估逻辑
- `GPT_Predict_Core.py`: LLM 辅助预测逻辑
- `llm_provider.py`: OpenAI / Gemini / Anthropic / mock 统一适配层
- `trace_web_export.py`: 将预测结果导出到 `Trace_Web`

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

主线 smoke 测试：

```bash
python main.py --max-pred-step 2
```

如果使用本地现成环境，可直接指定解释器：

```bash
D:\Workspace\1\wenyan-web\venv\Scripts\python.exe main.py --max-pred-step 2
```

## Trace_Web

前端目录：`Trace_Web/`

```bash
cd Trace_Web
npm install
npm run build
```

前端支持三类数据：

- demo 样本
- 实际样本
- 模型输出样本

## 说明

- 根目录 `.gitignore` 已忽略日志、评估图和派生 PNG
- 当前仓库已清理明显无关或无法运行的实验残留脚本
- 无可用 API Key 时，LLM 调用会自动回退到 `mock` 模式
>>>>>>> develop
