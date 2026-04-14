# HuggingFace Hub 上传二进制文件被拒绝问题

## 问题现象

使用 `hf upload` 上传包含 `.ply` 二进制文件的文件夹时，部分文件上传失败，报错：

```
Bad request for commit endpoint:
Your push was rejected because it contains binary files.
Please use https://huggingface.co/docs/hub/xet to store binary files.
Offending files:
  - assets/gs/scenes/arrange_flowers/flower2.ply
  - assets/gs/scenes/arrange_flowers/vase1.ply
  - assets/gs/scenes/stack_color_blocks/cube_yellow.ply
  - assets/gs/scenes/wipe_the_table/rake.ply
```

特征：同一批 `.ply` 文件中，大文件（>10MB）上传成功，小文件（<10MB）上传失败。

## 原因分析

`hf upload` 上传流程：

1. 调用 HF Hub 的 **preupload API**，服务端根据仓库的 `.gitattributes` 判断哪些文件走 LFS/Xet 通道，哪些文件作为普通文件内联提交。
2. 仓库默认的 `.gitattributes` 中**没有 `*.ply` 的 LFS 规则**。
3. 大文件超过 HF 的内置阈值，自动走 LFS 上传，所以成功。
4. 小文件（<10MB）被判定为普通文件，以内联方式提交。但 HF 服务端检测到内联内容是二进制数据，**拒绝提交**。

根本原因：**仓库远端的 `.gitattributes` 缺少对 `.ply` 文件类型的 LFS tracking 规则。**

> 注意：本地 `.gitattributes` 的修改不会影响 `hf upload` 的行为，因为 `hf upload` 使用 HTTP API 而非本地 git，preupload 判断依据的是远端仓库的 `.gitattributes`。

## 解决办法

### 步骤 1：在 `.gitattributes` 中添加 LFS 规则

```
*.ply filter=lfs diff=lfs merge=lfs -text
```

### 步骤 2：先单独上传 `.gitattributes` 到仓库

```bash
hf upload <repo_id> .gitattributes .gitattributes --repo-type=dataset
```

这样远端仓库就知道 `.ply` 文件应该走 LFS/Xet 通道。

### 步骤 3：再上传数据文件

```bash
hf upload <repo_id> assets/gs assets/gs --repo-type=dataset
```

此时所有 `.ply` 文件（无论大小）都会通过 LFS/Xet 上传，不再被拒绝。

## 举一反三

遇到其他二进制格式（如 `.obj`、`.stl`、`.glb` 等）上传被拒，同样在 `.gitattributes` 中添加对应规则即可：

```
*.obj filter=lfs diff=lfs merge=lfs -text
*.stl filter=lfs diff=lfs merge=lfs -text
*.glb filter=lfs diff=lfs merge=lfs -text
```

**关键：必须先将 `.gitattributes` 上传到远端仓库，再上传数据文件。**
