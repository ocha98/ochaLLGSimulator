## これは何？
LLG（Landau–Lifshitz–Gilbert）方程式を解くだけのシンプルなPythonスクリプトです。

自習用に作っているため仕様は頻繁に変わる可能性があります。🛠️

## 現状できること

| 機能                | 備考 |
|---------------------|------|
| マクロスピン LLG    | - |
| 一軸異方性軸は任意方向  | デフォルト z-軸（PMA）|
| Matplotlib で 3D 軌道＆時系列プロット自動生成 | `simulation_result.jpg` が出力されます |
| Mumax³ と数値一致   | 歳差運動／減衰／PMA で検証済み 

## 📦 必要環境

必要なモジュールをインストールしておいてください。

```bash
$ pip install numpy scipy matplotlib
```
