import type { Source } from './api'

/** CSS color for a 1-5 star rating, rounded to the nearest star. */
export function starColor(rating: number | undefined): string {
  if (rating == null) return 'var(--color-listing)'
  const r = Math.min(5, Math.max(1, Math.round(rating)))
  return `var(--color-star-${r})`
}

export function sourceColor(s: Source): string {
  return s.kind === 'review' ? starColor(s.meta.rating) : 'var(--color-listing)'
}

export const fmtInt = (n: number) => n.toLocaleString('en-US')
export const fmtPrice = (n: number) => `$${n.toFixed(2)}`
export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`)

/** Turn [3] citations into markdown links the renderer can swap for chips. */
export function linkCitations(text: string): string {
  return text.replace(/\[(\d{1,2})\](?!\()/g, '[$1](#src-$1)')
}
