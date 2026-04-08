# Docsify Cover Color Notes

This note captures a few practical patterns used when improving the homepage cover in `docs/index.html`.

## Goals

The original cover already used a dark gradient, but it still had a few common problems:

- the background felt visually flat
- bright accent colors were a bit too harsh against the dark base
- title, subtitle, body text, and CTA button did not yet form a clear visual hierarchy

The goal was to improve readability first, then add a bit more depth and polish.

## Useful Pattern: Promote Colors To Variables

Define cover-related colors in `:root` first:

```css
:root {
  --theme-color: #2f6fed;
  --cover-bg-top: #08111f;
  --cover-bg-mid: #10243f;
  --cover-bg-bottom: #1f4d78;
  --cover-title: #f8fbff;
  --cover-subtitle: #9fd7ff;
  --cover-text: #dce8f5;
  --cover-muted: #b7c9dd;
  --cover-button-bg: #ffd166;
  --cover-button-text: #10243f;
}
```

Why this helps:

- background and text colors stay coordinated
- later tuning becomes much easier
- accent colors can be changed without hunting through many selectors

## Useful Pattern: Build Depth With Layered Backgrounds

Instead of a single gradient, combine several layers:

```css
section.cover.show {
  background:
    radial-gradient(circle at 18% 22%, rgba(115, 197, 255, 0.28) 0%, rgba(115, 197, 255, 0) 32%),
    radial-gradient(circle at 82% 18%, rgba(255, 209, 102, 0.16) 0%, rgba(255, 209, 102, 0) 28%),
    linear-gradient(145deg, var(--cover-bg-top) 0%, var(--cover-bg-mid) 48%, var(--cover-bg-bottom) 100%);
}
```

This works well because:

- the linear gradient provides the base mood
- soft radial highlights create depth without becoming distracting
- the page feels more designed without requiring heavy graphics

## Useful Pattern: Add Subtle Overlay Texture

Pseudo-elements are a good way to enrich the cover background without touching the content markup.

Example ideas:

- a faint radial glow with `::before`
- a very light repeated line pattern with `::after`

The key is restraint. These layers should be barely noticeable and must not compete with the text.

## Useful Pattern: Put Copy On A Soft Surface

Wrapping the cover content visually with a translucent panel improved readability a lot:

```css
section.cover .cover-main {
  padding: 2.5rem 2rem;
  border-radius: 28px;
  background: linear-gradient(180deg, rgba(9, 19, 34, 0.32) 0%, rgba(9, 19, 34, 0.14) 100%);
  box-shadow: 0 22px 60px rgba(2, 10, 24, 0.28);
  backdrop-filter: blur(4px);
}
```

Why this helps:

- text contrast becomes more stable across the background
- the cover content becomes visually grouped
- the page gains depth without looking noisy

## Text Hierarchy Rules That Worked Well

### Title

Use the brightest color and strongest weight:

```css
section.cover .cover-main > h1,
section.cover .cover-main > h1 a {
  color: var(--cover-title);
  font-weight: 800;
}
```

### Subtitle

Use a cool accent color rather than strong yellow:

```css
section.cover .cover-main > blockquote > p {
  color: var(--cover-subtitle);
  font-weight: 600;
}
```

This gave emphasis without the harder contrast of saturated gold on dark blue.

### Body Text

Do not use pure white for everything. Slightly softened light text usually reads better:

```css
section.cover .cover-main > p {
  color: var(--cover-text);
}

section.cover .cover-main ul li {
  color: var(--cover-muted);
}
```

This creates separation between key lines and supporting lines.

## CTA Button Contrast

The CTA button worked better when it used:

- a warm highlight color for the fill
- a dark text color for strong contrast
- a soft shadow in the same hue family

Example:

```css
section.cover .cover-main > p:last-child a {
  background-color: var(--cover-button-bg);
  color: var(--cover-button-text);
  border-color: var(--cover-button-bg);
}
```

On hover, keep the palette related to the default state. A transparent hover is fine, but the text and border should still remain easy to see.

## Practical Guidelines

- Use one dominant hue family for the background and one accent family for emphasis
- Reserve the brightest color for the most important element
- Keep body text slightly softer than the title
- Treat subtitle and CTA as different emphasis levels, not the same level
- Prefer gentle shadows over high-opacity black shadows
- When using a dark cover, test whether yellow accents are too sharp before committing

## Rule Of Thumb

When tuning a Docsify homepage cover, improve it in this order:

1. background depth
2. text contrast
3. hierarchy between title, subtitle, and body
4. CTA visibility
5. decorative texture

If readability becomes worse at any step, simplify.
