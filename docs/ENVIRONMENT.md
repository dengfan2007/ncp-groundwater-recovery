# 环境与复现范围

2026-10-17 聚焦叙事修订新增 `analysis/ncp_story_analysis.py` 入口，与原日历和评审证据共同支持当前稿。仍使用既有Anaconda Python；仅将 EMD-signal 1.10.0（无依赖升级）安装在 `outputs/runs/2026-10-16-story-runtime-r1/`，通过 `--emd-runtime` 指定。该目录被Git忽略，可按 `STORY_REVISION_2026-10-17.md` 中的固定版本命令重建；wheel哈希与运行设定已记录。未修改全局包、Earth Engine认证或项目配置。

2026-10-12 新分析使用既有 `D:/ProgramData/anaconda/python.exe`（Python 3.11.7）。实际导入 pandas 2.3.3、numpy 1.26.4、scipy 1.11.4、statsmodels 0.14.0、Rbeast 0.1.25、matplotlib 3.8.0。环境中的 pandas distribution metadata 仍报2.1.4，当前运行清单同时记录两者；未安装或升级软件。复现当前稿使用 `analysis/ncp_calendar_revision.py`，证据见 `outputs/calendar_revision/2026-10-12-r4/manifest.json`，不要沿用旧流水线生成最新正文。

研究分析旧清单：`outputs/repro_manifest.json`，记录日期 2026-04-17。该文件保留原文中的旧绝对路径以反映历史运行，不是本次整理的重新运行报告。

旧环境为 Python 3.11.7（Anaconda），主要包版本：pandas 2.1.4、numpy 1.26.4、statsmodels 0.14.0、scipy 1.11.4、Rbeast 0.1.25、earthengine-api 1.7.22、matplotlib 3.8.0、geopandas 1.1.3、shapely 2.1.2、openpyxl 3.1.5。

`analysis/requirements-historical.txt` 仅转录这个已知清单，不声称包含全部传递依赖或能在任何系统中直接安装成功。原 `analysis/requirements.txt` 保留，其 pandas/scipy 最低版本高于旧实际运行版本；在科学重跑前需明确采用历史环境还是经过验证的新环境。

本次管理工具只使用 Python 标准库。正文编译使用本机 XeLaTeX 和 BibTeX；本次没有安装或升级软件，没有重新认证 GEE，没有重跑研究分析。

不要将 GEE 服务账户私钥或 OAuth 凭据放进 Git。原第三方井网仓库的内容已复制，嵌套 `.git` 未复制；URL 和原提交号记在迁移清单中，不改变全局 Git safe.directory。

图和 PDF 的历史编译结果可以保留；修改 TeX 后必须重新编译，不能把未更新的预览当作新版本。
