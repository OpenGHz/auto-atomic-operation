# Plan: Migrate assets/gs from Git LFS to Hugging Face

## Context
`assets/gs/` (291 MB, ~23 PLY files) is stored via Git LFS. We will move it to Hugging Face Hub, purge from git history, and document the download command. `assets/meshes/` stays in LFS unchanged.

---

## Step 1: Upload assets to Hugging Face

Repo: https://huggingface.co/datasets/OpenGHz/auto-atom-assets

Small PLY files get rejected because the remote `.gitattributes` lacks a `*.ply` LFS rule. Fix: upload `.gitattributes` first, then upload data (see `docs/huggingface-binary-upload-fix.md`).

```bash
# 1a. Create a .gitattributes with PLY LFS rule
echo '*.ply filter=lfs diff=lfs merge=lfs -text' > /tmp/hf_gitattributes

# 1b. Upload .gitattributes to remote repo first
hf upload OpenGHz/auto-atom-assets /tmp/hf_gitattributes .gitattributes --repo-type=dataset

# 1c. Now upload assets/gs/ (all PLY files go through LFS/Xet)
hf upload OpenGHz/auto-atom-assets assets/gs assets/gs --repo-type=dataset
```

## Step 2: Update `.gitattributes`

Remove only this line:
```
assets/gs/** filter=lfs diff=lfs merge=lfs -text
```

Keep `assets/meshes/**` and `assets/videos/**` LFS lines.

## Step 3: Add `.gitignore` entry

Append `assets/gs/` to `.gitignore`.

## Step 4: Purge `assets/gs/` from git history

```bash
pip install git-filter-repo
cp -r .git .git-backup

git filter-repo --invert-paths --path assets/gs/ --force

git remote add origin git@github.com:OpenGHz/auto-atomic-operation.git
git push origin --force --all
git push origin --force --tags
```

> **Warning**: Rewrites all commit hashes. Collaborators must re-clone.

## Step 5: Update README.md

Replace LFS pull instructions for GS assets with:
```markdown
### Download GS Assets
pip install huggingface_hub
hf download OpenGHz/auto-atom-assets --repo-type=dataset --include "assets/gs/*" --local-dir .
```

---

## Files to modify
| File | Action |
|------|--------|
| `.gitattributes` | Remove 1 LFS line (`assets/gs/**`) |
| `.gitignore` | Add `assets/gs/` |
| `README.md` | Update download instructions |

## Verification
1. Fresh clone -> `hf download OpenGHz/auto-atom-assets --repo-type=dataset --include "assets/gs/*" --local-dir .` -> verify `assets/gs/` populated
2. `python examples/compare_gs_render.py` -- verify GS PLY loading
3. `git log --all --full-history -- assets/gs/` -- confirm no history remains
