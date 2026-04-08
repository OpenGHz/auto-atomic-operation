# Docsify / GitHub Pages GIF Notes

This note summarizes a few practical pitfalls we hit when serving GIF assets in the Docsify site from `docs/`.

## Symptom

Local preview works with:

```bash
npx docsify-cli serve docs
```

But after deploying `docs/` to GitHub Pages, GIFs on the homepage do not display.

## Root Causes

### 1. `docs/` deployment only publishes files inside `docs/`

If a markdown file in `docs/` references:

```md
![demo](assets/videos/demo.gif)
```

then GitHub Pages can only serve it if `docs/assets/videos/demo.gif` is an actual published file.

If `docs/assets/videos` is a symlink to `../../assets/videos`, local preview may still work, but GitHub Pages deployment is not a reliable place to depend on that structure.

### 2. Git LFS changes how direct links behave

This repository tracks:

```gitattributes
assets/videos/** filter=lfs diff=lfs merge=lfs -text
```

That means common direct-link patterns may not return the real GIF bytes:

- `raw.githubusercontent.com/...` often returns an LFS pointer text file
- `cdn.jsdelivr.net/gh/...` may also resolve to the LFS pointer instead of the actual binary

This makes an `<img>` render fail even though the URL looks correct.

## What Worked

For LFS-hosted GIFs, this form worked:

```md
![pick_and_place](https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/pick_and_place.gif)
```

Compared with the other options:

- `raw.githubusercontent.com` returned `text/plain` and only about 131 bytes
- `jsDelivr` returned a tiny LFS-pointer-sized response instead of the full GIF
- `media.githubusercontent.com` returned the real `image/gif` payload

## Useful Checks

When a GIF does not show on GitHub Pages, check the response headers directly.

Example:

```bash
curl -I -L https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/pick_and_place.gif
```

Healthy signs:

- `content-type: image/gif`
- a realistic `content-length`

Bad signs:

- `content-type: text/plain`
- `content-length: 131` or another tiny pointer-sized response

## Cache Gotchas

Even after fixing the link, GitHub Pages and browser caches may continue showing the old result for a while.

Practical tricks:

- wait a few minutes after deployment
- hard refresh the page
- add a simple cache-busting query string such as `?v=20260407`

Example:

```md
![pick_and_place](https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/pick_and_place.gif?v=20260407)
```

## Recommended Rule Of Thumb

Use one of these two strategies and avoid mixing temporary hacks:

### Option A. Most stable for GitHub Pages

Place display assets as real files under `docs/`, for example:

```text
docs/assets/videos/demo.gif
```

Then reference them with a docs-relative path:

```md
![demo](./assets/videos/demo.gif)
```

This is the most predictable static-site setup.

### Option B. Accept external hosting

If the source asset stays in Git LFS outside `docs/`, use:

```text
https://media.githubusercontent.com/media/<owner>/<repo>/<branch>/assets/videos/<file>.gif
```

This avoids duplicating the GIF inside `docs/`, but introduces dependence on external hosting and caching behavior.

## Short Checklist

- If GitHub Pages deploys from `docs/`, make sure page assets are actually reachable from the published site
- Do not assume symlinks that work locally will behave the same after Pages deployment
- For Git LFS media, do not assume `raw.githubusercontent.com` or `jsDelivr` returns the actual binary
- Verify suspicious links with `curl -I -L`
- Expect cache delay after changing media URLs
