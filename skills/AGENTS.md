# Skills 目录规范

`skills/` 是本仓库维护技能的唯一实体目录. 每个技能位于 `skills/<name>/`, 入口文件为 `SKILL.md`.

- Claude 项目入口 `.claude/skills/<name>` 和 Codex 项目入口 `.codex/skills/<name>` 必须使用相对软链接 `../../skills/<name>`.
- Codex 用户级入口 `~/.codex/skills/<name>` 必须使用绝对软链接指向仓库中的 `skills/<name>`.
- 不在工具目录直接创建或复制本仓库维护的技能. 工具自带和第三方技能不纳入此目录.
