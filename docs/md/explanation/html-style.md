# HTML 教程与调研页样式说明

本文作为自有 HTML 教程与调研页的说明层落点. 它解释这类 HTML 页面为什么存在, 如何与 Markdown 分工, 以及应该先看什么.

## 这是什么

这是对 `learn/` 与 `research/` 下 HTML 教程与调研页的说明页. 它不单独定义框架行为, 也不取代样式规范正文.

## 为什么需要

`docs/md/` 和 `learn/`, `research/` 下的 HTML 页面职责不同:

- `docs/md/`: 事实源和实现导航
- `learn/`, `research/` 下的 HTML: 阅读体验和概念呈现

需要单独一页把这层分工讲清楚, 否则容易把 HTML 教程或调研页误当成唯一事实源.

## 核心分工

- `docs/md/` 按架构 / 说明 / 使用三层承载正文, 是行为, 字段, API 和兼容边界的事实源
- `learn/` 与 `research/` 下的 HTML 页面只把同一套事实讲得更易读, 不定义新契约
- 同一主题变化时, 先更新 Markdown, 再同步对应 HTML 页面

## 该看哪里

- 看正式样式规范: [../architecture/html-style-policy.md](../architecture/html-style-policy.md)
- 看兼容锚点: [../README.md#xdl-html-阅读页样式规范](../README.md#xdl-html-阅读页样式规范)

## 常见误区

- 不要让 HTML 单独定义新契约
- 不要把阅读页和操作手册混写
- 不要复制第三套视觉系统到新页面里