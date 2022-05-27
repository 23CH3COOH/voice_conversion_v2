# voice_conversion_v2

## 概要
WORLDを用いた統計的声質変換

## 必要環境
pythonとrequirements.txtにあるライブラリがインストールされていること

## 実行方法
name1の音声をname2に変換する場合

1. wav/train/にフォルダname1/とname2/を作り、学習データとなるname1の音声ファイルとname2の音声ファイルを格納する
    - 内容は同じでなければならない
    - 同じ内容の音声でファイル名を同じにする
2. wav/production/にフォルダname1/を作り、name2の声に変換したいname1の音声ファイルを格納する
3. make_converter.pyでConverterMakerのコンストラクタの第1引数に'name1'を、第2引数に'name2'を指定してrun()を実行
4. convert_voice.pyでmain関数の第1引数に'name1'を、第2引数に'name2'を指定して実行
5. wav/production/name2/に、wav/production/name1/にある音声ファイルからname2の声に変換したものが出力される

## その他
- 全体の流れを説明した図をimageフォルダに格納