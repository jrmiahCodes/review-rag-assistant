import { useState } from 'react'
import type { Source } from '../api'
import { fmtInt, fmtPrice, sourceColor } from '../lib'

function Stars({ rating }: { rating: number }) {
  const r = Math.round(rating)
  return (
    <span className="num inline-flex items-center gap-[2px]" aria-label={`${r} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <svg key={i} viewBox="0 0 10 10" className="h-[11px] w-[11px]" aria-hidden>
          <path
            d="M5 .6l1.3 2.9 3.1.3-2.4 2.1.7 3.1L5 7.4 2.3 9l.7-3.1L.6 3.8l3.1-.3z"
            fill={i <= r ? 'currentColor' : 'none'}
            stroke="currentColor"
            strokeWidth=".8"
          />
        </svg>
      ))}
    </span>
  )
}

function ScoreBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100
  return (
    <div className="flex items-center gap-2 text-xs text-ink-faint">
      <span className="w-14 shrink-0">{label}</span>
      <span className="relative h-1.5 w-20 overflow-hidden rounded-full bg-ground">
        <span className="absolute inset-y-0 left-0 rounded-full bg-ink-soft" style={{ width: `${pct}%` }} />
      </span>
      <span className="num w-9 text-ink-soft">{value.toFixed(2)}</span>
    </div>
  )
}

function SourceItem({ s, highlighted, flash }: { s: Source; highlighted: boolean; flash: boolean }) {
  const [open, setOpen] = useState(false)
  const color = sourceColor(s)
  const attached = s.kind === 'product' && s.rerank_score == null && s.score === 0

  return (
    <li
      id={`src-${s.n}`}
      className={`relative scroll-mt-24 border-b border-line bg-surface py-3.5 pr-4 pl-5 last:border-b-0 ${flash ? 'flash' : ''} ${
        highlighted ? 'bg-[color-mix(in_srgb,var(--color-action)_6%,white)]' : ''
      }`}
    >
      <span className="absolute inset-y-0 left-0 w-1" style={{ backgroundColor: color }} aria-hidden />
      <div className="flex items-start gap-3">
        <span
          className="num mt-0.5 flex h-5 min-w-5 shrink-0 items-center justify-center rounded px-1 text-[11px] font-bold text-white"
          style={{ backgroundColor: color }}
        >
          {s.n}
        </span>
        <div className="min-w-0 flex-1">
          {s.kind === 'review' ? (
            <>
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs" style={{ color }}>
                <Stars rating={s.meta.rating ?? 0} />
                <span className="text-ink-faint">
                  Review{s.meta.review_date ? `, ${s.meta.review_date.slice(0, 7)}` : ''}
                  {s.meta.verified_purchase ? ', verified purchase' : ''}
                </span>
              </div>
              <p className="mt-1 line-clamp-2 text-[13px] text-ink-faint">{s.title}</p>
              {s.meta.review_title && <p className="mt-1.5 text-sm font-semibold">{s.meta.review_title}</p>}
              <p className={`mt-1 text-sm leading-relaxed text-ink-soft ${open ? '' : 'line-clamp-4'}`}>{s.text}</p>
              {s.text.length > 260 && (
                <button
                  type="button"
                  onClick={() => setOpen(!open)}
                  className="mt-1 text-xs font-semibold text-action hover:text-action-ink"
                >
                  {open ? 'Show less' : 'Show full review'}
                </button>
              )}
            </>
          ) : (
            <>
              <p className="text-xs text-ink-faint">
                {attached ? 'Listing for a reviewed product' : 'Product listing'}
                {s.meta.platform ? `, ${s.meta.platform}` : ''}
              </p>
              <p className="mt-1 line-clamp-2 text-sm font-semibold">{s.title}</p>
              <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                {s.meta.price != null && (
                  <span className="num text-lg font-bold tracking-tight">{fmtPrice(s.meta.price)}</span>
                )}
                {s.meta.average_rating != null && (
                  <span className="num text-sm text-ink-soft">
                    {s.meta.average_rating.toFixed(1)} stars from {fmtInt(s.meta.rating_number ?? 0)} ratings
                  </span>
                )}
              </div>
            </>
          )}
          {!attached && (
            <div className="mt-2.5 space-y-1">
              {s.rerank_score != null && <ScoreBar label="Rerank" value={s.rerank_score} />}
              <ScoreBar label="Search" value={s.score} />
            </div>
          )}
        </div>
      </div>
    </li>
  )
}

export function Sources({ sources, hovered, flashed }: { sources: Source[]; hovered: number | null; flashed: number | null }) {
  if (!sources.length) return null
  return (
    <ol className="overflow-hidden rounded-[10px] border border-line">
      {sources.map((s) => (
        <SourceItem key={`${s.n}-${s.id}`} s={s} highlighted={hovered === s.n} flash={flashed === s.n} />
      ))}
    </ol>
  )
}
