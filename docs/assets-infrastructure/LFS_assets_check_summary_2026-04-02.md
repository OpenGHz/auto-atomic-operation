# auto-atomic-operation assets Git LFS 排查总结

日期：2026-04-02
仓库：OpenGHz/auto-atomic-operation（branch: main）

## 1. 问题现象

- assets 目录中大量文件曾表现为很小（约 130B）。
- 随后出现大量 Git 状态修改（约 573 个 M）。

### 1.1 复现与初始观察命令

目的：确认当前仓库、LFS 工具和状态规模。

```bash
cd /home/haizhou/auto-atomic-operation
pwd
git rev-parse --is-inside-work-tree
git lfs version
git status --short | wc -l
git status --short | head -n 20
```

预期：

- 能看到仓库路径和 `true`。
- 能看到 git-lfs 版本。
- `git status` 可量化当前异常规模（例如 500+ 个 `M`）。

## 2. 结论

本次不是文件损坏，而是 Git LFS 指针文件与工作区实体文件状态切换后，索引元数据未同步导致的状态异常。

- 小文件阶段：是 LFS pointer（正常现象，表示实体尚未 hydrate）。
- 下载后阶段：实体文件已恢复正常，但索引仍保留旧的 pointer stat 信息，导致 Git 显示大量 M。

## 3. 关键证据

- assets 下 pointer 签名文件统计为 0（不再是 pointer 文本）。
- Git LFS 完整性检查通过：Git LFS fsck OK。
- 抽样文件校验通过：
  - 指针中 oid sha256 与实体文件 sha256 一致（match=yes）。
  - 示例文件 flower1.obj：工作区约 2.35MB，且内容为 OBJ 实体文本头，不是 pointer。
- Git 低层状态显示 .M，但普通 diff 为空，符合“索引 stat 缓存不同步”特征。

### 3.1 判断是否仍是 LFS pointer 文本

目的：确认工作区里是否还残留“130B 指针文件”。

```bash
cd /home/haizhou/auto-atomic-operation
rg -l '^version https://git-lfs.github.com/spec/v1$' assets | wc -l
rg -l '^version https://git-lfs.github.com/spec/v1$' assets | head -n 20
```

预期：

- 若输出为 `0`，说明 assets 中已无 pointer 文本文件。
- 若大于 `0`，说明仍有未 hydrate 文件，需要继续 `git lfs pull`。

### 3.2 拉取 LFS 实体并观察进度

目的：把 pointer 文件替换为真实对象内容。

```bash
cd /home/haizhou/auto-atomic-operation
git lfs pull
```

预期：

- 出现下载进度。
- 完成后，抽样文件大小明显大于 130B。

### 3.3 校验 LFS 对象完整性

目的：排除“对象下载后损坏”。

```bash
cd /home/haizhou/auto-atomic-operation
git lfs fsck
```

预期：

- 输出 `Git LFS fsck OK`。
- 若报错，优先排查网络中断、对象缺失、磁盘问题。

### 3.4 抽样校验 pointer OID 与实体文件哈希一致性

目的：确认“下载实体文件”与“指针声明对象”一致。

```bash
cd /home/haizhou/auto-atomic-operation
f='assets/meshes/arrange_flowers/flower1/flower1.obj'
ptr_oid=$(git show :"$f" | awk '/^oid sha256:/{sub("oid sha256:","",$0);print}')
real_oid=$(sha256sum "$f" | awk '{print $1}')
printf 'pointer_oid=%s\nreal_oid=%s\nmatch=%s\n' "$ptr_oid" "$real_oid" "$( [[ "$ptr_oid" = "$real_oid" ]] && echo yes || echo no )"
```

预期：

- `match=yes`。
- 若 `match=no`，需立即重拉该对象并检查缓存目录权限。

### 3.5 排除“仍有异常小文件”

目的：快速发现可疑残留文件。

```bash
cd /home/haizhou/auto-atomic-operation
find assets -type f -size -200c | wc -l
find assets -type f -size -200c | head -n 30
```

预期：

- 通常应为 `0`。

### 3.6 定位为何 Git 仍显示大量 M

目的：区分真实内容变化 vs 索引/元数据不同步。

```bash
cd /home/haizhou/auto-atomic-operation
f='assets/meshes/arrange_flowers/flower1/flower1.obj'

# 低层状态
git status --porcelain=v2 -- "$f"

# 索引中的 blob 内容（通常是 pointer）
git show :"$f" | sed -n '1,5p'

# 工作区实体内容（通常是模型真实文本头）
sed -n '1,5p' "$f"

# 关键：比较 hash（过滤前/过滤后）
git ls-files --stage -- "$f"
git hash-object --no-filters "$f"
git hash-object "$f"

# 查看索引缓存的 stat 信息
git ls-files --debug -- "$f" | sed -n '1,20p'
stat -c 'size=%s mode=%a mtime=%Y' "$f"
```

判断逻辑：

- `git hash-object --no-filters` 与 `git hash-object` 不同：说明 clean/smudge 过滤在正常工作。
- `git show :"$f"` 是 pointer、工作区是实体：属于 LFS 正常形态。
- 若 `git diff -- "$f"` 为空但 `status` 仍是 `.M`：通常是索引 stat 未刷新。

## 4. 根因说明

LFS 跟踪文件在索引中存的是 pointer blob；工作区是 smudge 后实体内容。
当索引仍记录旧的文件大小/时间戳（如 132B pointer 时期）而工作区已是实体内容时，Git 可能持续显示未暂存修改，即使文件内容哈希校验是正确的。

## 5. 已执行修复

执行：git add -- assets

结果：

- 修改计数从约 572/573 降为 0。
- staged 内容差异数为 0（未引入真实内容变更，只是刷新索引元数据）。

### 5.1 修复命令与验证命令（可直接复用）

```bash
cd /home/haizhou/auto-atomic-operation

before=$(git status --short | wc -l)
git add -- assets
after=$(git status --short | wc -l)

printf 'before=%s\nafter=%s\n' "$before" "$after"

# 验证是否有真实 staged 内容变更
git diff --cached --name-only | wc -l
```

预期：

- `after` 显著下降（本次为 0）。
- `git diff --cached --name-only | wc -l` 为 `0`。

说明：

- 该操作用于刷新索引状态，不会凭空修改实体内容。
- 若你只想修复部分目录，可改为 `git add -- assets/meshes/...`。

## 6. 推荐排查流程（后续可复用）

1. 先看规模：`git status --short | wc -l`。
2. 查是否仍为 pointer：`rg -l '^version https://git-lfs.github.com/spec/v1$' assets`。
3. 如仍有 pointer，执行：`git lfs pull`。
4. 完整性检查：`git lfs fsck`。
5. 抽样一致性检查：pointer oid vs `sha256sum`。
6. 若 `diff` 为空但大量 `.M`：执行 `git add -- <lfs_path>` 刷新索引。
7. 最后确认：`git status --short` 为干净。

## 7. 最终状态

- assets 实体文件正常。
- 无损坏证据。
- 仓库状态已恢复干净。
