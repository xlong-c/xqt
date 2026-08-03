# HTML 阅读页样式说明

本文作为自有 HTML 阅读页的说明层落点. 它解释 HTML 阅读页为什么存在, 如何与 Markdown 分工, 以及应该先看什么.

## 这是什么

这是对 `docs/html/` 视觉阅读层的说明页. 它不单独定义框架行为, 也不取代样式规范正文.

## 为什么需要

`docs/html/` 和 `docs/md/` 的职责不同:

- `docs/md/`: 事实源和实现导航
- `docs/html/`: 阅读体验和概念呈现

需要单独一页把这层分工讲清楚, 否则容易把 HTML 阅读页误当成唯一事实源.

## 核心分工

- Markdown 按架构 / 说明 / 使用三层承载正文
- HTML 负责把同一套事实讲得更易读
- 同一主题变化时, 先更新 Markdown, 再同步 HTML

## 该看哪里

- 看正式样式规范: [../architecture/html-style-policy.md](../architecture/html-style-policy.md)
- 看阅读入口: [../../html/index.html](../../html/index.html)
- 看样式展示: [../../html/style-showcase.html](../../html/style-showcase.html)

## 常见误区

- 不要让 HTML 单独定义新契约
- 不要把阅读页和操作手册混写
- 不要复制第三套视觉系统到新页面里
