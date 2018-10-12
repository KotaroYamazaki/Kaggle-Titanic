# Kaggle-Titanic
## eda/
  - EDA(Explanatory Data Analysis): 探索的データ解析
## tutorial_qiita/
  - Qiita の記事をもとにやってみた
## data/
  - 入力データ(公式サイトより)
## prediction/
- randamforest.py: ランダムフォレストによる分類、パラメータ推定にてGridSearchを利用
- validation.ipynb: 数種類のモデルに対して訓練データを分割しそれぞれを評価
- voting.py: ランダムフォレスト, SVM, NBの組み合わせを決め多数決で出力する

### Best Score
0.79425

### 環境
  - Jupter Notebook 4.1.0
  - Python 3.5.1
  - Anaconda 4.0.0
