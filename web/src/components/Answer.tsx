import Markdown from 'react-markdown'
import type { Source } from '../api'
import { linkCitations, sourceColor } from '../lib'

type Props = {
  text: string
  streaming: boolean
  sources: Source[]
  onCite: (n: number) => void
  onHover: (n: number | null) => void
}

export function Answer({ text, streaming, sources, onCite, onHover }: Props) {
  const byN = new Map(sources.map((s) => [s.n, s]))

  return (
    <div className={`prose-answer ${streaming ? 'caret' : ''}`}>
      <Markdown
        components={{
          a: ({ href, children }) => {
            const m = href?.match(/^#src-(\d+)$/)
            if (!m) return <a href={href}>{children}</a>
            const n = Number(m[1])
            const src = byN.get(n)
            if (!src) {
              // The model cited a number that doesn't exist: show it, flagged, rather than hide it.
              return (
                <span
                  className="mx-0.5 inline-block rounded border border-dashed border-star-1 px-1 text-[0.75em] font-semibold text-star-1"
                  title="This citation doesn't match any retrieved source"
                >
                  {n}?
                </span>
              )
            }
            const color = sourceColor(src)
            return (
              <button
                type="button"
                onClick={() => onCite(n)}
                onMouseEnter={() => onHover(n)}
                onMouseLeave={() => onHover(null)}
                onFocus={() => onHover(n)}
                onBlur={() => onHover(null)}
                className="num mx-[1px] inline-flex h-[1.35em] min-w-[1.5em] -translate-y-[0.1em] items-center justify-center rounded px-1 align-middle text-[0.72em] font-bold text-white transition-transform hover:-translate-y-[0.2em]"
                style={{ backgroundColor: color }}
                aria-label={`Source ${n}: ${src.kind === 'review' ? `${src.meta.rating}-star review` : 'product listing'} of ${src.title}`}
              >
                {n}
              </button>
            )
          },
        }}
      >
        {linkCitations(text)}
      </Markdown>
    </div>
  )
}
