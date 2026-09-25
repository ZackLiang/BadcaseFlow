# GitHub 发布检查清单

这份清单用于把本地仓库整理成第一个可公开使用的版本。

## 1. 检查目录

确认仓库根目录是 BadcaseFlow：

```bash
git rev-parse --show-toplevel
git status --short
```

确认下面这些内容存在：

```text
README.md
LICENSE
CHANGELOG.md
CONTRIBUTING_zh.md
pyproject.toml
src/badcaseflow/
tests/
examples/
docs/
.github/workflows/ci.yml
```

## 2. 本地验收

```bash
python -m pip install -e .
python -m badcaseflow version
python -m badcaseflow doctor --root . --strict
python -m badcaseflow demo --run runs/release-demo
python -m badcaseflow data prepare-gsm8k \
  --input examples/datasets/gsm8k_tiny.jsonl \
  --output-dir runs/release-gsm8k
python -m compileall src tests
python -m unittest discover -s tests
git diff --check
```

如果要验证 parquet 输出：

```bash
python -m pip install -e ".[parquet]"
python -m badcaseflow data prepare-gsm8k \
  --input examples/datasets/gsm8k_tiny.jsonl \
  --output-dir runs/release-gsm8k-parquet \
  --parquet
```

## 3. 检查仓库内容

```bash
git status --short
git diff --stat
git diff -- README.md pyproject.toml
```

不要提交以下本地运行目录：

```text
runs/
artifacts/
checkpoints/
.badcaseflow/
```

## 4. 提交到 GitHub

第一次提交：

```bash
git add .
git commit -m "feat: prepare BadcaseFlow v0.1"
git branch -M main
git remote -v
git push -u origin main
```

后续迭代：

```bash
git add .
git commit -m "docs: improve project documentation"
git push
```

## 5. GitHub 页面检查

上传后建议在 GitHub 页面确认：

- README 首屏可以看到项目定位、安装命令和最小 demo；
- `docs/autodl_gsm8k_quickstart_zh.md` 可以正常打开；
- Actions 中的 CI 通过；
- examples 目录中的 recipe 和数据文件可以直接点击查看；
- LICENSE、CHANGELOG 和贡献指南已经显示；
- 仓库根目录没有本地运行产物。
