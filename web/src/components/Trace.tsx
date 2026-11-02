import type { ToolCall, Timings } from '../api'
import { fmtMs } from '../lib'

const TOOL_LABEL: Record<string, string> = {
  search_products: 'Searched product listings',
  search_reviews: 'Searched customer reviews',
}

function Args({ args }: { args: Record<string, unknown> }) {
  const entries = Object.entries(args).filter(([, v]) => v !== null && v !== '' && v !== undefined)
  return (
    <dl className="mt-1.5 flex flex-wrap gap-1.5">
      {entries.map(([k, v]) => (
        <div key={k} className="flex overflow-hidden rounded border border-line text-xs">
          <dt className="bg-ground px-1.5 py-0.5 text-ink-faint">{k}</dt>
          <dd className="px-1.5 py-0.5 text-ink">{typeof v === 'string' ? `"${v}"` : String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

const STAGES: { key: keyof Timings; label: string; color: string }[] = [
  { key: 'route', label: 'Route', color: '#8a90a6' },
  { key: 'retrieve', label: 'Search + rerank', color: '#3b45c7' },
  { key: 'generate', label: 'Write answer', color: '#1a2033' },
]

function TimingBar({ t }: { t: Timings }) {
  const total = STAGES.reduce((a, s) => a + t[s.key], 0) || 1
  return (
    <div>
      <div className="flex h-2.5 overflow-hidden rounded-full" role="img" aria-label={`Total ${fmtMs(t.total)}`}>
        {STAGES.map((s) => (
          <span key={s.key} style={{ width: `${(t[s.key] / total) * 100}%`, backgroundColor: s.color }} />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-soft">
        {STAGES.map((s) => (
          <li key={s.key} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: s.color }} aria-hidden />
            {s.label} <span className="num text-ink">{fmtMs(t[s.key])}</span>
          </li>
        ))}
        <li>
          First word after <span className="num text-ink">{fmtMs(t.route + t.retrieve + t.first_token)}</span>
        </li>
      </ul>
    </div>
  )
}

type Props = {
  calls: ToolCall[]
  timings: Timings | null
  model: string | null
  usage: { input_tokens: number; output_tokens: number } | null
  settings: string
}

export function Trace({ calls, timings, model, usage, settings }: Props) {
  if (!calls.length) return null
  return (
    <details className="group mt-8 rounded-[10px] border border-line bg-surface" open>
      <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm font-semibold select-none">
        How this was answered
        <svg viewBox="0 0 12 12" className="h-3 w-3 text-ink-faint transition-transform group-open:rotate-180" aria-hidden>
          <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      </summary>
      <div className="space-y-5 border-t border-line px-4 pt-4 pb-5 text-sm">
        <section>
          <h3 className="text-xs text-ink-faint">The router turned your question into</h3>
          <ul className="mt-2 space-y-3">
            {calls.map((c, i) => (
              <li key={i}>
                <span className="font-semibold">{TOOL_LABEL[c.name] ?? c.name}</span>
                <Args args={c.arguments} />
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h3 className="text-xs text-ink-faint">Retrieval</h3>
          <p className="mt-1 text-ink-soft">{settings}</p>
        </section>
        {timings && (
          <section>
            <h3 className="mb-2 text-xs text-ink-faint">Time spent</h3>
            <TimingBar t={timings} />
          </section>
        )}
        {model && usage && (
          <p className="text-xs text-ink-faint">
            Written by <span className="text-ink">{model}</span> from{' '}
            <span className="num">{usage.input_tokens.toLocaleString()}</span> input tokens,{' '}
            <span className="num">{usage.output_tokens.toLocaleString()}</span> output tokens.
          </p>
        )}
      </div>
    </details>
  )
}
