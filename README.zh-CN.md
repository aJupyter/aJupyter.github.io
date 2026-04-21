<div align="center">

# Luka Homepage

### 一个温暖、极简、响应式的个人主页模板，适合研究者、学生、开发者和创作者。

[English README](README.md) · [在线预览](https://wzsyyh.github.io/luka-homepage-template/) · [使用此模板](https://github.com/wzsyyh/luka-homepage-template/generate)

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-c2714f?style=flat-square)](https://wzsyyh.github.io/luka-homepage-template/)
[![Template Ready](https://img.shields.io/badge/Template-Ready-3d3929?style=flat-square)](https://github.com/wzsyyh/luka-homepage-template/generate)
[![Stack](https://img.shields.io/badge/Stack-HTML%20%7C%20CSS%20%7C%20JS-e28a67?style=flat-square)](https://github.com/wzsyyh/luka-homepage-template)
[![License](https://img.shields.io/badge/License-MIT-8c8577?style=flat-square)](./LICENSE.md)

</div>

`Luka Homepage` 是从我自己的个人主页中抽离出来的一个独立模板版本。它追求温暖、克制、学术友好的视觉风格，同时保持纯静态、零构建、易部署。

## 为什么用它

- 极简但不单调：更重视字体、配色、留白和阅读节奏。
- 学术场景友好：不是通用作品集拼贴，而是为个人主页内容定制。
- 零构建成本：直接部署到 GitHub Pages 即可。
- 易于复用：改少量文案和资源就能快速上线。
- 已做好移动端适配：桌面端和手机端都能直接使用。

## 模板包含什么

| 模块 | 内容 |
| --- | --- |
| 个人信息 | `About Me`、头像、邮箱、社交链接 |
| 学术时间线 | `Education`、`Experience` |
| 展示区 | `Projects`、`Publications`、`Awards` |
| 体验细节 | 深色模式、轻量动画、响应式布局 |
| 可替换资源 | 头像、机构 Logo、favicon、CV 链接 |

## 快速开始

### 1. 创建你自己的版本

直接使用 GitHub 的 [Use This Template](https://github.com/wzsyyh/luka-homepage-template/generate)，或者把这个目录整体复制到一个新仓库中。

推荐仓库名：

- `luka-homepage`
- `luka-homepage-template`
- `luka-academic-homepage`

### 2. 修改内容

通常只需要编辑这些文件：

- `index.html`：页面结构与主要内容
- `assets/css/theme-luka.css`：颜色、间距和布局
- `assets/img/avatar.svg`：头像占位图
- `assets/img/institution.svg`：机构 Logo 占位图

### 3. 替换资源

- 头像：替换 `assets/img/avatar.svg`
- 机构 Logo：替换 `assets/img/institution.svg`
- favicon：替换 `assets/img/favicon.svg`
- 简历：修改 `index.html` 中的 `Download my CV` 链接

### 4. 部署到 GitHub Pages

1. 打开仓库 `Settings`
2. 进入 `Pages`
3. 选择 `Deploy from a branch`
4. 选择 `main` 和 `/ (root)`
5. 保存并等待部署完成

## 目录结构

```text
.
├── index.html
├── README.md
├── README.zh-CN.md
├── LICENSE.md
└── assets
    ├── css
    │   ├── font_sans_serif.css
    │   └── theme-luka.css
    ├── cv
    ├── img
    └── js
        └── scale.fix.js
```

## 设计说明

- 左侧栏默认保持顶部对齐，和原始个人主页布局一致。
- 视觉语言更偏温暖、克制，而不是炫技式装饰。
- 模板保留了足够的学术主页结构，同时又尽量降低了个性化门槛。

## 署名与版权

默认页脚包含两行：

1. 你的站点版权信息
2. `Luka Homepage Template by Yuheng Yang`

如果你希望保留模板来源，建议保留第二行。

## 许可证

这个模板自带 `LICENSE.md`，可以直接作为独立仓库发布。
