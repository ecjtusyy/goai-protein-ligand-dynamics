# GOAI Protein–Ligand Dynamics

GOAI「小分子–蛋白质结合轨迹预测」初赛实验。

当前目标：先在 MISATO_100 上读通真实轨迹，再依次实现 Static、Linear、Residual MLP 和多步几何约束模型。

## 数据

- 官方数据：MISATO `MD.hdf5`
- 开发子集：NeuralMD 作者提供的 `MISATO_100`
- 数据文件不上传 GitHub

## 进度

- [ ] 读取单条 ligand 轨迹
- [ ] 构造 T1 / T2 / T3
- [ ] Static 与 Linear baseline
- [ ] RMSD–horizon 曲线
- [ ] Residual MLP
- [ ] Multi-step + geometry
