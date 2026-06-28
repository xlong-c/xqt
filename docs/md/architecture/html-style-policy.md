# XDL HTML 阅读页规范

本文承接 `XDL` 自有 HTML 阅读页的规范正文. 它定义长期 HTML 页面应该怎样组织, 引用什么公共资源, 以及哪些写法不允许继续扩散.

## 负责什么

- 定义 `docs/html/` 自有阅读页的长期规范.
- 定义公共 CSS, 主题脚本和 body 模板的使用边界.
- 定义 HTML 与 Markdown 的分工和同步规则.

## 不负责什么

- 不替代具体 HTML 页面内容本身.
- 不单独解释为什么需要阅读层概念页.
- 不承载训练, 配置或 API 行为事实.

## 第一原则

- `docs/md/` 是事实源和实现导航.
- `docs/html/` 是给人类读者的可视化阅读层.
- 同一主题变化时, 先更新对应 Markdown 正文, 再同步 HTML.
- HTML 可以提炼, 重排和图文化 Markdown, 但不能单独定义新契约.

## 公共资源边界

`docs/html/assets/xdl-doc.css` 是仓库自有 HTML 阅读页的唯一公共样式入口. 它负责:

- 主题 token, 强调色, 深浅色和旧变量别名
- 基础排版, 链接, 表格, 代码块, 图片, 打印和响应式行为
- 通用页面组件, 例如 `.topbar`, `.hero`, `.button`, `.card`, `.section`, `.layout`, `.toc`, `.article`, `.callout`, `.note`, `.table-wrap`, `.flow`
- 双栏长文骨架等可复用版式
- `body.xdl-style-atlas`, `body.xdl-style-ledger` 两种项目级版式模板

目录专属 CSS 只允许作为薄入口和局部扩展:

- 第一行必须 `@import` 对应相对路径的 `xdl-doc.css`
- 只写带目录或页面命名空间的局部组件
- 不复制 reset, 主题变量, 字体栈, topbar, hero, layout, toc, card, table, callout 或通用阅读组件
- 同一组件被两个以上目录需要时, 先提升到 `xdl-doc.css`
- 不为单页新增一个只改颜色, 间距或卡片样式的 CSS 文件

## 页面资源引用

`docs/html/*.html` 默认写法:

```html
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>页面标题</title>
  <link rel="stylesheet" href="assets/xdl-doc.css">
  <script defer src="assets/xdl-theme.js"></script>
</head>
```

需要目录专属 CSS 时, 该 CSS 顶部必须先导入统一样式:

```css
@import "../../../docs/html/assets/xdl-doc.css";
```

## body 模板

项目自有长期 HTML 必须且只能使用两种版式之一:

- `xdl-style-atlas`: 面向入口页, 导航页, 速查页和交互实验页
- `xdl-style-ledger`: 面向长文, 教程, 调研报告和结构说明页

页面通过 body class 选择版式:

```html
<body class="xdl-style-atlas">
```

```html
<body class="xdl-style-ledger">
```

目录专属页面可以叠加语义 class, 例如:

```html
<body class="xdl-style-ledger math-doc-page">
<body class="xdl-style-ledger research-page">
<body class="xdl-style-atlas flash-attention-page">
```

这类语义 class 只表达页面语义或局部组件命名空间, 不能单独出现在 body 上, 也不能复制成第三套主题系统.

## 页面骨架

页面结构优先使用这个基础骨架:

```html
<body>
  <a class="skip-link" href="#main">跳到正文</a>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="index.html">XDL Docs</a>
      <nav aria-label="页面导航">
        <a href="#overview">概览</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div>
      <p class="eyebrow">Reading Page</p>
      <h1>页面标题</h1>
      <p class="lead">一句话说明页面解决什么问题.</p>
    </div>
  </section>

  <main id="main">
    <section id="overview" class="section">
      <div class="section-head">
        <h2>概览</h2>
        <p>先给读者主线, 再进入细节.</p>
      </div>
    </section>
  </main>
</body>
```

## 主题和 token

统一 CSS 支持主题:

- `system`
- `light`
- `dark`
- `sepia`

统一 CSS 支持强调色:

- `teal`
- `blue`
- `violet`
- `amber`
- `rose`
- `green`

主题状态写在根元素:

```html
<html data-theme="dark" data-accent="blue">
```

新增或重构自有 HTML/CSS 时优先使用这些 token:

```css
--xdl-page
--xdl-surface
--xdl-surface-soft
--xdl-surface-muted
--xdl-text
--xdl-muted
--xdl-muted-strong
--xdl-border
--xdl-border-strong
--xdl-accent
--xdl-accent-strong
--xdl-accent-soft
--xdl-code-bg
--xdl-code-border
--xdl-code-text
--xdl-radius
--xdl-shadow
--xdl-content
--xdl-reading
```

## 版式约束

- 阅读主宽度控制在 `--xdl-reading` 到 `--xdl-content` 之间
- 长正文不要铺满超宽屏
- 页面要有明确的 `header` 和正文 `main`
- 表格必须可横向滚动
- 卡片只用于并列信息块, 工具面板, 索引项和局部容器
- 圆角默认 `8px`
- 不使用单一色相铺满全页
- 交互元素必须有 hover / focus 可见状态

## 可访问性和维护

- 页面必须保留 `<meta name="viewport" content="width=device-width, initial-scale=1">`
- 长页建议加 `.skip-link` 和目录
- 图片和图示要有 `alt` 或 `aria-label`
- 不要用颜色作为唯一信息来源
- 主题切换后仍要检查代码块, 表格, 按钮和 callout 的对比度
- 打印时隐藏导航, 目录和主题按钮, 正文保持可读

## 新增页面检查清单

- 已引用 `xdl-doc.css`, 没有复制整套内联样式
- body 已且仅已选择 `xdl-style-atlas` 或 `xdl-style-ledger` 之一
- 没有 `<style>` 块和散落的 `style=`
- 需要主题按钮时已引用 `xdl-theme.js`
- 使用统一 token, 没有大面积硬编码颜色
- 在浅色, 深色和窄屏下结构不重叠
- 表格和公式块可横向滚动
- HTML 内容和对应 Markdown / 源码事实一致

## 相关页面

- [xdl.md](xdl.md)
- [../explanation/html-style.md](../explanation/html-style.md)
- [../README.md#xdl-html-阅读页样式规范](../README.md#xdl-html-阅读页样式规范)
- [../../html/index.html](../../html/index.html)
