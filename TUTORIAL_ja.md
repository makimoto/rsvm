# RSVM チュートリアル: インストールからモデル訓練まで

このチュートリアルでは、RSVM（Rust Support Vector Machine）ライブラリの使用方法について、インストールから実際のデータセットでのモデル訓練・評価まで、段階的にガイドします。

## 目次

1. [インストール](#インストール)
2. [クイックスタート](#クイックスタート)
3. [CLI使用方法](#CLI使用方法)
4. [データフォーマットの使い方](#データフォーマットの使い方)
5. [初回モデル訓練](#初回モデル訓練)
6. [モデル評価](#モデル評価)
7. [高度な設定](#高度な設定)
8. [実世界での例](#実世界での例)
9. [パフォーマンスのコツ](#パフォーマンスのコツ)
10. [トラブルシューティング](#トラブルシューティング)

## インストール

### 前提条件

- Rust 1.70 以降
- Cargo パッケージマネージャ

### オプション1: 依存関係として追加

`Cargo.toml`にRSVMを追加：

```toml
[dependencies]
rsvm = "0.1.0"
```

### オプション2: ソースからビルド

```bash
git clone https://github.com/your-org/rsvm.git
cd rsvm
cargo build --release
```

CLI バイナリをビルドする場合：

```bash
cargo build --bin rsvm --release
```

### インストール確認

```bash
cargo test
```

すべてのテストが通れば、インストールが成功しています。

CLIの動作確認：

```bash
./target/release/rsvm --help
```

## クイックスタート

### 簡単な例

```rust
use rsvm::api::SVM;
use rsvm::{Sample, SparseVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 訓練データを作成
    let samples = vec![
        Sample::new(SparseVector::new(vec![0, 1], vec![2.0, 1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![1.8, 1.1]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-2.0, -1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.8, -1.1]), -1.0),
    ];

    // モデルを訓練
    let model = SVM::new()
        .with_c(1.0)
        .train_samples(&samples)?;

    // 予測を実行
    let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![1.5, 0.8]), 1.0);
    let prediction = model.predict(&test_sample);
    
    println!("予測ラベル: {}", prediction.label);
    println!("信頼度: {:.3}", prediction.confidence());

    Ok(())
}
```

## CLI使用方法

RSVMには、モデルの訓練、予測、評価を簡単に行えるコマンドラインインターフェースが含まれています。

### 基本的なCLIコマンド

```bash
# ヘルプの表示
rsvm --help

# 利用可能な全コマンドを表示
rsvm train --help
rsvm predict --help
rsvm evaluate --help
rsvm info --help
```

### モデル訓練

```bash
# 基本的な訓練（LibSVMフォーマット）
rsvm train --data training_data.libsvm --output my_model.json

# CSVデータでの訓練
rsvm train --data training_data.csv --output my_model.json --format csv

# カスタムパラメータでの訓練
rsvm train --data training_data.libsvm --output my_model.json \
    -C 10.0 --epsilon 0.0001 --max-iterations 2000

# 詳細出力付きの訓練
rsvm train --data training_data.libsvm --output my_model.json --verbose
```

### モデル情報の確認

```bash
# 保存されたモデルの詳細情報を表示
rsvm info my_model.json
```

出力例：
```
=== SVM Model Summary ===
Kernel Type: linear
Support Vectors: 23
Bias: 0.123456
Library Version: 0.1.0
Created: 2025-06-12T10:30:00+00:00
Training Parameters:
  C: 1
  Epsilon: 0.001
  Max Iterations: 1000
```

### クイック操作

```bash
# 訓練/テスト分割での評価
rsvm quick eval training_data.libsvm test_data.libsvm

# クロスバリデーション
rsvm quick cv data.libsvm --ratio 0.8

# カスタムパラメータでのクイック評価
rsvm quick cv data.libsvm --ratio 0.8 -C 5.0
```

### 実践的なCLI使用例

LibSVMデータでの完全なワークフロー：

```bash
# 1. サンプルデータの作成
cat > sample_data.libsvm << EOF
+1 1:2.0 2:1.0
+1 1:1.8 2:1.1
+1 1:2.2 2:0.9
-1 1:-2.0 2:-1.0
-1 1:-1.8 2:-1.1
-1 1:-2.2 2:-0.9
EOF

# 2. モデル訓練
rsvm train --data sample_data.libsvm --output trained_model.json --verbose

# 3. モデル情報確認
rsvm info trained_model.json

# 4. クロスバリデーション
rsvm quick cv sample_data.libsvm --ratio 0.8
```

### パラメータチューニング

```bash
# 異なるC値でのバッチ評価
for c in 0.1 1.0 10.0 100.0; do
  echo "Testing C=$c"
  rsvm quick cv data.libsvm -C $c --ratio 0.8
done
```

### 現在の制限事項

**注意**: 現在のCLI実装では、以下の機能は制限があります：

- `predict` コマンド：保存済みモデルからの予測機能（プレースホルダー実装）
- `evaluate` コマンド：保存済みモデルでの評価機能（プレースホルダー実装）

これらの機能は、モデル再構築機能が実装された後に完全に利用可能になります。現在は `quick` コマンドを使用してリアルタイムで訓練・評価を行うことを推奨します。

詳細なCLI使用例については、[CLI_EXAMPLES.md](CLI_EXAMPLES.md) を参照してください。

## データフォーマットの使い方

RSVMは2つの人気のあるデータフォーマットをサポートしています：LibSVMとCSVです。

### LibSVMフォーマット

LibSVMフォーマットは機械学習で広く使用されています：

```
+1 1:2.0 2:1.0 3:0.5
-1 1:-2.0 2:-1.0 3:-0.5
+1 2:1.5 4:2.0
```

フォーマット: `<ラベル> <インデックス>:<値> <インデックス>:<値> ...`

- ラベル: +1（正クラス）または -1（負クラス）
- インデックス: 1ベースの特徴量インデックス
- 値: 特徴量の値（非ゼロ値のみ指定が必要）

#### LibSVMデータの作成

```rust
// data.libsvm
use std::fs::File;
use std::io::Write;

let mut file = File::create("data.libsvm")?;
writeln!(file, "+1 1:2.0 2:1.0")?;
writeln!(file, "-1 1:-2.0 2:-1.0")?;
writeln!(file, "+1 1:1.8 2:1.1")?;
writeln!(file, "-1 1:-1.8 2:-1.1")?;
```

#### LibSVMデータの読み込み

プログラム内で：
```rust
use rsvm::api::SVM;

let model = SVM::new()
    .with_c(1.0)
    .train_from_file("data.libsvm")?;
```

CLIで：
```bash
rsvm train --data data.libsvm --output model.json
```

### CSVフォーマット

CSVフォーマットは表計算ソフトやデータベースからのデータに便利です：

```csv
feature1,feature2,label
2.0,1.0,1
-2.0,-1.0,-1
1.8,1.1,1
-1.8,-1.1,-1
```

- 最後の列: ラベル（自動的に+1/-1に変換）
- その他の列: 特徴量
- ヘッダー: 自動検出

#### CSVデータの読み込み

プログラム内で：
```rust
use rsvm::api::SVM;

let model = SVM::new()
    .with_c(1.0)
    .train_from_csv("data.csv")?;
```

CLIで：
```bash
rsvm train --data data.csv --output model.json --format csv
```

## 初回モデル訓練

RSVMでモデル訓練を始める2つの方法があります：プログラム内でのAPI使用とCLIツールの使用です。

### CLI を使った簡単な訓練

CLIを使って手軽にモデル訓練を始めましょう：

```bash
# サンプルデータ作成
cat > first_training.libsvm << EOF
+1 1:2.0 2:2.0
+1 1:2.1 2:1.9
+1 1:1.9 2:2.1
-1 1:-2.0 2:-2.0
-1 1:-2.1 2:-1.9
-1 1:-1.9 2:-2.1
EOF

# モデル訓練
rsvm train --data first_training.libsvm --output first_model.json --verbose

# モデル情報確認
rsvm info first_model.json

# クロスバリデーション
rsvm quick cv first_training.libsvm --ratio 0.8
```

### プログラム内でのモデル訓練

#### ステップ1: データの準備

線形分離可能な簡単なデータセットを作成しましょう：

```rust
use std::fs::File;
use std::io::Write;

fn create_sample_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("tutorial_data.libsvm")?;
    
    // 正クラス: (2, 2)周辺の点
    writeln!(file, "+1 1:2.0 2:2.0")?;
    writeln!(file, "+1 1:2.1 2:1.9")?;
    writeln!(file, "+1 1:1.9 2:2.1")?;
    writeln!(file, "+1 1:2.2 2:1.8")?;
    
    // 負クラス: (-2, -2)周辺の点
    writeln!(file, "-1 1:-2.0 2:-2.0")?;
    writeln!(file, "-1 1:-2.1 2:-1.9")?;
    writeln!(file, "-1 1:-1.9 2:-2.1")?;
    writeln!(file, "-1 1:-2.2 2:-1.8")?;
    
    Ok(())
}
```

#### ステップ2: モデルの訓練

```rust
use rsvm::api::SVM;

fn train_model() -> Result<(), Box<dyn std::error::Error>> {
    // サンプルデータを作成
    create_sample_data()?;
    
    // カスタムパラメータで訓練
    let model = SVM::new()
        .with_c(1.0)                    // 正則化パラメータ
        .with_epsilon(0.001)            // 収束許容誤差
        .with_max_iterations(1000)      // 最大反復回数
        .train_from_file("tutorial_data.libsvm")?;
    
    println!("モデル訓練完了！");
    println!("サポートベクター数: {}", model.info().n_support_vectors);
    println!("バイアス: {:.3}", model.info().bias);
    
    Ok(())
}
```

#### ステップ3: モデルのテスト

```rust
use rsvm::{Sample, SparseVector};

fn test_model(model: &rsvm::api::TrainedModel<rsvm::LinearKernel>) {
    let test_cases = vec![
        (vec![1.5, 1.5], "正クラスであるべき"),
        (vec![-1.5, -1.5], "負クラスであるべき"),
        (vec![0.0, 0.0], "境界ケース"),
    ];
    
    for (coords, description) in test_cases {
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1], coords),
            0.0  // 未知ラベル
        );
        
        let prediction = model.predict(&test_sample);
        println!("{}: {} (信頼度: {:.3})", 
                description, 
                prediction.label, 
                prediction.confidence());
    }
}
```

## モデル評価

### CLI での評価

現在のクイック評価：
```bash
# クロスバリデーション
rsvm quick cv tutorial_data.libsvm --ratio 0.8

# 別々の訓練・テストファイルでの評価
rsvm quick eval train_data.libsvm test_data.libsvm

# パラメータを変えての評価
rsvm quick cv tutorial_data.libsvm --ratio 0.8 -C 5.0
```

### プログラム内での評価

#### 基本的な精度

```rust
// 訓練データでの評価（確認用）
let accuracy = model.evaluate_from_file("tutorial_data.libsvm")?;
println!("訓練精度: {:.1}%", accuracy * 100.0);
```

#### 詳細メトリクス

```rust
use rsvm::LibSVMDataset;

let dataset = LibSVMDataset::from_file("tutorial_data.libsvm")?;
let metrics = model.evaluate_detailed(&dataset);

println!("精度: {:.3}", metrics.accuracy());
println!("適合率: {:.3}", metrics.precision());
println!("再現率: {:.3}", metrics.recall());
println!("F1スコア: {:.3}", metrics.f1_score());
println!("特異度: {:.3}", metrics.specificity());
```

#### クロスバリデーション

```rust
use rsvm::api::quick;

// シンプルな訓練/テスト分割バリデーション
let accuracy = quick::simple_validation(&dataset, 0.8, 1.0)?;
println!("クロスバリデーション精度: {:.1}%", accuracy * 100.0);
```

## 高度な設定

### パラメータチューニング

```rust
// 異なるCの値を試す
let c_values = vec![0.1, 1.0, 10.0, 100.0];

for &c in &c_values {
    let model = SVM::new()
        .with_c(c)
        .train_from_file("tutorial_data.libsvm")?;
    
    let accuracy = model.evaluate_from_file("tutorial_data.libsvm")?;
    println!("C = {}: 精度 = {:.1}%", c, accuracy * 100.0);
}
```

### メモリ最適化

```rust
// 大きなデータセットの場合、キャッシュサイズを増加
let model = SVM::new()
    .with_c(1.0)
    .with_cache_size(100 * 1024 * 1024)  // 100MBキャッシュ
    .train_from_file("large_dataset.libsvm")?;
```

### カスタムカーネル

```rust
use rsvm::kernel::LinearKernel;

// 線形カーネルを明示的に使用
let model = SVM::with_kernel(LinearKernel::new())
    .with_c(1.0)
    .train_from_file("data.libsvm")?;
```

## 実世界での例

より現実的なデータセット - 二値分類用に適応した古典的なIrisデータセットで作業してみましょう。

### Irisデータの準備

```rust
use std::fs::File;
use std::io::Write;

fn create_iris_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("iris_binary.csv")?;
    
    // ヘッダー
    writeln!(file, "sepal_length,sepal_width,petal_length,petal_width,class")?;
    
    // Setosa（クラス1）vs その他（クラス-1）
    // Setosaサンプル
    writeln!(file, "5.1,3.5,1.4,0.2,1")?;
    writeln!(file, "4.9,3.0,1.4,0.2,1")?;
    writeln!(file, "4.7,3.2,1.3,0.2,1")?;
    writeln!(file, "4.6,3.1,1.5,0.2,1")?;
    writeln!(file, "5.0,3.6,1.4,0.2,1")?;
    
    // 非Setosaサンプル（Versicolor/Virginica）
    writeln!(file, "7.0,3.2,4.7,1.4,-1")?;
    writeln!(file, "6.4,3.2,4.5,1.5,-1")?;
    writeln!(file, "6.9,3.1,4.9,1.5,-1")?;
    writeln!(file, "5.5,2.3,4.0,1.3,-1")?;
    writeln!(file, "6.5,2.8,4.6,1.5,-1")?;
    
    Ok(())
}

fn iris_classification_example() -> Result<(), Box<dyn std::error::Error>> {
    create_iris_data()?;
    
    // モデルを訓練
    let model = SVM::new()
        .with_c(10.0)  // このデータセットには高いCを使用
        .train_from_csv("iris_binary.csv")?;
    
    // 評価
    let accuracy = model.evaluate_from_csv("iris_binary.csv")?;
    println!("Iris分類精度: {:.1}%", accuracy * 100.0);
    
    // 新しいサンプルでテスト
    let test_setosa = Sample::new(
        SparseVector::new(vec![0, 1, 2, 3], vec![5.2, 3.4, 1.4, 0.2]),
        0.0
    );
    
    let test_versicolor = Sample::new(
        SparseVector::new(vec![0, 1, 2, 3], vec![6.8, 3.0, 4.8, 1.4]),
        0.0
    );
    
    println!("テストSetosa予測: {}", model.predict(&test_setosa).label);
    println!("テストVersicolor予測: {}", model.predict(&test_versicolor).label);
    
    Ok(())
}
```

## パフォーマンスのコツ

### 1. データ前処理

```rust
// 数値安定性のため、特徴量を正規化
fn normalize_features(samples: &mut [Sample]) {
    // 各特徴量のmin/maxを求める
    // min-max正規化を適用: (x - min) / (max - min)
    // 実装は具体的なデータに依存
}
```

### 2. スパースデータ

```rust
// スパースベクターを効率的に使用
let sparse_sample = Sample::new(
    SparseVector::new(
        vec![5, 100, 1000],      // 非ゼロインデックスのみ指定
        vec![1.5, 2.0, 0.8]      // 対応する値
    ),
    1.0
);
```

### 3. バッチ操作

```rust
// 複数サンプルにはバッチ予測を使用
let predictions = model.predict_batch(&test_samples);
```

### 4. メモリ管理

```rust
// 大きなデータセットのメモリ使用量を推定
use rsvm::utils::memory;

let n_samples = 10000;
let estimated_memory = memory::estimate_kernel_cache_memory(n_samples);
println!("推定メモリ使用量: {} MB", estimated_memory / (1024 * 1024));

// 適切なキャッシュサイズを設定
let cache_size = memory::recommend_cache_size(n_samples, 1000 * 1024 * 1024); // 1GB利用可能
```

## トラブルシューティング

### よくある問題

**1. 精度が悪い**
```rust
// 異なるCの値を試す
let c_values = vec![0.01, 0.1, 1.0, 10.0, 100.0];
// データ品質とバランスを確認
// 特徴量スケーリングを検討
```

**2. 訓練が遅い**
```rust
// 初期テスト用にmax_iterationsを減らす
let model = SVM::new()
    .with_max_iterations(100)  // テスト用に低く設定
    .train_samples(&samples)?;

// 大きなデータセット用にキャッシュサイズを増加
let model = SVM::new()
    .with_cache_size(500 * 1024 * 1024)  // 500MB
    .train_samples(&samples)?;
```

**3. メモリ問題**
```rust
// 非常に大きなデータセットの場合、キャッシュサイズを削減
let model = SVM::new()
    .with_cache_size(50 * 1024 * 1024)   // 50MB
    .train_samples(&samples)?;
```

**4. 収束問題**
```rust
// epsilonとmax_iterationsを調整
let model = SVM::new()
    .with_epsilon(0.01)        // より緩い収束条件
    .with_max_iterations(2000) // より多い反復
    .train_samples(&samples)?;
```

### エラーメッセージ

- `EmptyDataset`: データファイルが存在し、有効なサンプルが含まれていることを確認
- `ParseError`: データフォーマットを確認（LibSVMインデックスは1ベース、CSV最後の列がラベル）
- `InvalidParameter`: C > 0、epsilon > 0、max_iterations > 0であることを確認

### デバッグのコツ

```rust
// デバッグ出力を有効化
let info = model.info();
println!("サポートベクター数: {}", info.n_support_vectors);
println!("サポートベクターインデックス: {:?}", info.support_vector_indices);

// データ読み込みを確認
let dataset = LibSVMDataset::from_file("data.libsvm")?;
println!("{}サンプル、{}次元を読み込み", dataset.len(), dataset.dim());
```

## 完全な例プログラム

全工程を示す完全な例です：

```rust
use rsvm::api::{SVM, quick};
use rsvm::{LibSVMDataset, Sample, SparseVector};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RSVMチュートリアル例");
    println!("==================");
    
    // 1. サンプルデータ作成
    create_tutorial_dataset()?;
    
    // 2. モデル訓練
    println!("\n1. モデル訓練中...");
    let model = SVM::new()
        .with_c(1.0)
        .with_epsilon(0.001)
        .train_from_file("tutorial_dataset.libsvm")?;
    
    println!("   ✓ モデル訓練完了");
    println!("   サポートベクター数: {}", model.info().n_support_vectors);
    
    // 3. モデル評価
    println!("\n2. モデル評価中...");
    let accuracy = model.evaluate_from_file("tutorial_dataset.libsvm")?;
    println!("   精度: {:.1}%", accuracy * 100.0);
    
    // 4. 詳細メトリクス
    let dataset = LibSVMDataset::from_file("tutorial_dataset.libsvm")?;
    let metrics = model.evaluate_detailed(&dataset);
    println!("   適合率: {:.3}", metrics.precision());
    println!("   再現率: {:.3}", metrics.recall());
    println!("   F1スコア: {:.3}", metrics.f1_score());
    
    // 5. 予測テスト
    println!("\n3. 予測テスト中...");
    test_predictions(&model);
    
    // 6. クロスバリデーション
    println!("\n4. クロスバリデーション中...");
    let cv_accuracy = quick::simple_validation(&dataset, 0.8, 1.0)?;
    println!("   クロスバリデーション精度: {:.1}%", cv_accuracy * 100.0);
    
    println!("\n✓ チュートリアル完了！");
    
    Ok(())
}

fn create_tutorial_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("tutorial_dataset.libsvm")?;
    
    // 線形分離可能な2Dデータセットを作成
    let positive_samples = vec![
        (2.0, 2.0), (2.1, 1.9), (1.9, 2.1), (2.2, 1.8),
        (1.8, 2.2), (2.3, 2.0), (2.0, 2.3), (1.7, 1.9),
    ];
    
    let negative_samples = vec![
        (-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1), (-2.2, -1.8),
        (-1.8, -2.2), (-2.3, -2.0), (-2.0, -2.3), (-1.7, -1.9),
    ];
    
    for (x, y) in positive_samples {
        writeln!(file, "+1 1:{} 2:{}", x, y)?;
    }
    
    for (x, y) in negative_samples {
        writeln!(file, "-1 1:{} 2:{}", x, y)?;
    }
    
    Ok(())
}

fn test_predictions(model: &rsvm::api::TrainedModel<rsvm::LinearKernel>) {
    let test_cases = vec![
        ((1.5, 1.5), "正領域"),
        ((-1.5, -1.5), "負領域"),
        ((0.5, 0.5), "境界近く（正側）"),
        ((-0.5, -0.5), "境界近く（負側）"),
        ((0.0, 0.0), "原点"),
    ];
    
    for ((x, y), description) in test_cases {
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1], vec![x, y]),
            0.0
        );
        
        let prediction = model.predict(&test_sample);
        println!("   {} ({}, {}): {} (信頼度: {:.3})", 
                description, x, y, 
                if prediction.label > 0.0 { "+" } else { "-" },
                prediction.confidence());
    }
}
```

このチュートリアルは、基本的なインストールから高度な使用法まですべてをカバーしています。RSVMライブラリを使って強力なSVMモデルを構築できるようになりました！

より多くの例と高度な機能については、[APIドキュメント](docs/)と[examples](examples/)ディレクトリを確認してください。