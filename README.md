# 並列分散処理 2025年度
このリポジトリは広島大学情報科学部「並列分散処理」（2025年度）の演習で作成したプログラムと，レポートに掲載したグラフ生成用スクリプト・出力を管理する．  
レポート本文（.tex/.pdf）は含めていないが，演習で利用したすべてのソースと実行結果は格納している．  


本人であることを確認するために，学生番号(Bも含めて)と名前のハッシュ値を以下に記す．
テキストファイルに改行無しASCII互換文字で`BXXXXXX-FIRSTNAME-LASTNAME`の形式で，英大文字と数字で入力すると以下のハッシュ値が得られる．
ファイルを使わず`echo`を用いても同じ結果が得られる．
```
$ openssl dgst -sha256 a.txt
SHA2-256(a.txt)= 45b1545236a2f50e128ff12f245fc33bdb99b26f380ed254ae572881cca19d8b
$ echo 'BXXXXXX-FIRSTNAME-LASTNAME' | openssl dgst -sha256
SHA2-256(stdin)= 45b1545236a2f50e128ff12f245fc33bdb99b26f380ed254ae572881cca19d8b
```

このリポジトリは成績評価確定後に予告なくprivate化または削除される可能性がある．
レポートの.tex/.pdfファイルは含めていない．公開を継続する場合は成績評価確定後に追加する予定がある．

## 動作環境の目安
- g++ 14系，OpenMP
- OpenMPI 4系（演習2）
- CUDA 12系（演習3，GPU必須）
- Python 3系＋matplotlib（グラフ描画）
- ImageMagick

## 使い方（概要）
- `make -C 1` / `make -C 2` / `make -C 3` で各日程のバイナリを生成（1/Makefile, 2/Makefile, 3/Makefile）。
- グラフ描画スクリプト  
  - 1日目: `1/2img.py`, `1/3img.py`（OpenMP実行結果を読み込みEPS/PNGを生成）  
  - 2日目: `2/1img.py`, `2/2img.py`（MPI実行と計測・描画を一括実行）  
  - 3日目: カーネル内で計測し標準出力を保存（`3/*.stdout.txt`）；補助的な可視化は適宜追加
  - convert: PNGファイルをEPSファイルに変換する
- 実行例  
  - OpenMP: `make -C 1 3.out` などでビルドし，必要に応じて `OMP_NUM_THREADS` を設定して実行。  
  - MPI: `mpirun -np <プロセス数> ./2/1.out hypercube` のようにモードを指定。  
  - CUDA: `nvcc`でビルド済みの`3/*.out`を実行（GPU環境が必要）。


## /1
1日目（OpenMP）のソース・実行結果・グラフ生成スクリプト
## /2
2日目（MPIとNUMA実験）のソース・実行結果・グラフ生成スクリプト
## /3
3日目（CUDA）のソース・実行結果