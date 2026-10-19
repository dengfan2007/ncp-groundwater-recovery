# 图1制作源与使用说明

图1含区域水文地质定位图HLM（a）、详细研究区SAM（b）和固定井群（c）。主稿调用01_manuscript/Fig01.pdf；此目录另提供HLM.pdf、SAM.pdf及PNG预览。可编辑图为01_manuscript/Fig01.svg；GIS图层为EPSG:4326 GeoJSON。主绘图程序为项目scripts/build_hydrogeology_maps.py，原始下载程序为scripts/fetch_hydrogeology_map_inputs.py。

## 数据来源与许可

| 内容 | 来源及处理 | 使用依据 |
| --- | --- | --- |
| 区域含水系统类型 | BGR/UNESCO，Groundwater Resources of the World (WHYMAP)，2008，1:25,000,000；2026-10-19读取官方GIS服务第11层。仅按aquif_type合并补给量子类，裁切到图幅及陆地 | BGR/UNESCO允许注明来源后使用/重现；图注已署名。官方说明：https://www.bgr.bund.de/whymap/EN/Maps_Data/maps_data_node_en.html |
| 陆地/海岸、河流、陆地国界 | Natural Earth 1:10m。保留相关陆地国界；剔除overlay limits，将朝鲜半岛军事分界线单独以虚线表示；不绘海上国界 | 公有领域：https://www.naturalearthdata.com/about/terms-of-use/ |
| 低地轮廓 | 复用manuscripts/assets/jhrs-v30/figure_01_lowland_mask_200m.geojson，原记录为SRTM/GAUL2015生成的≤200m显示轮廓 | 沿用既有来源和资产，不作为新计算的分析掩膜；源文件含完整处理元数据 |
| 网格及井点 | 复用outputs/figure_revision/2026-10-17-final/tables/fig01_analysis_cells.csv和fig01_fixed_wells.csv | 既有研究输出：27格，北11/南16；共同18格；潜水井310、承压井284 |
| 国名与代码 | UN M49：https://unstats.un.org/unsd/methodology/m49/ | 国名和PRK/KOR代码已与下载的UN表核对 |
| 边界惯例参考 | UN Map of the World，No.4651 Rev.4，July 2026；图面中可见的相关陆地国界和邻接关系作区域视觉对照 | UN官网文件本次返回403/SSL错误，实际查看的是公开镜像中的UN署名原图；精确来源、哈希和限制见un_reference_check.json。未将UN原图重印到论文 |

Natural Earth各图层的下载地址、时间、SHA-256及成功/失败记录见download_manifest.json。完整下载文件保留在项目outputs/runs/2026-10-19-hj-map-inputs/；本目录的裁剪图层可直接导入QGIS。正文新增BGR/UNESCO和Natural Earth两条参考文献。

## 地图解释边界

- WHYMAP只用于区域水文地质背景；不应读取为局地含水层接触线、浅深层分界或井筛管层位，未绘制没有资料支持的流向、剖面或厚度。
- 37°N为分析分组线；六个37°N中心网格在主分组归南方。它与低地轮廓都不是含水层边界。
- b、c地图显示范围略宽于正文选区，以完整呈现原网格；a中的红框表示正文112–120°E、34.5–40.6°N选区框。
- 河流为概化的地理参照，不表示研究时段的实际流量、补水路径或因果作用。
- 城市、山脉和省名为地理标签；未把标注位置当作新的分析数据。

图件原始宽176mm、高155mm；矢量PDF嵌入Arial字体，PNG为600dpi。SVG提供矢量编辑入口；带白色衬边的部分标签以矢量轮廓保存，文字内容也可在绘图脚本中修改。
