import { useEffect, useRef, useState } from 'react'
import { ask, type AskOptions, type Meta, type Source, type Timings, type ToolCall } from '../api'
import { Answer } from '../components/Answer'
import { Sources } from '../components/Sources'
import { Trace } from '../components/Trace'
import { fmtInt } from '../lib'

const EXAMPLES = [
  'Do DualSense controllers get stick drift?',
  'Best rated Switch screen protector under $15',
  'Wireless headsets under $60 that people like',
  'Is Farming Simulator 20 on Switch the full game?',
  'What do reviewers think of the PS6 controller?',
]

const MODES: { value: AskOptions['mode']; label: string }[] = [
  { value: 'hybrid', label: 'Hybrid' },
  { value: 'vector', label: 'Vector' },
  { value: 'bm25', label: 'Keyword' },
]

type Run = {
  question: string
  calls: ToolCall[]
  sources: Source[]
  text: string
  streaming: boolean
  timings: Timings | null
  model: string | null
  usage: { input_tokens: number; output_tokens: number } | null
  error: string | null
  settings: string
}

export function AskPage({ meta }: { meta: Meta | null }) {
  const [question, setQuestion] = useState('')
  const [opts, setOpts] = useState<AskOptions>({ mode: 'hybrid', rerank: true })
  const [run, setRun] = useState<Run | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [flashed, setFlashed] = useState<number | null>(null)
  const abort = useRef<AbortController | null>(null)
  const busy = run?.streaming ?? false

  useEffect(() => () => abort.current?.abort(), [])

  async function submit(q: string) {
    const trimmed = q.trim()
    if (trimmed.length < 3 || busy) return
    setQuestion(trimmed)
    abort.current?.abort()
    abort.current = new AbortController()
    const modeLabel = MODES.find((m) => m.value === opts.mode)?.label ?? opts.mode
    setRun({
      question: trimmed,
      calls: [],
      sources: [],
      text: '',
      streaming: true,
      timings: null,
      model: null,
      usage: null,
      error: null,
      settings: `${modeLabel} search, ${opts.rerank ? '30 candidates reranked by a cross-encoder' : 'no reranking'}.`,
    })
    try {
      await ask(
        trimmed,
        opts,
        (e) =>
          setRun((r) => {
            if (!r) return r
            switch (e.type) {
              case 'plan':
                return { ...r, calls: e.calls }
              case 'sources':
                return { ...r, sources: e.sources }
              case 'token':
                return { ...r, text: r.text + e.text }
              case 'done':
                return { ...r, timings: e.timings_ms, model: e.model, usage: e.usage }
              case 'error':
                return { ...r, error: e.message }
            }
          }),
        abort.current.signal,
      )
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setRun((r) =>
          r && {
            ...r,
            error: `Couldn't reach the backend. Start it with "uv run rra serve" and try again. (${(err as Error).message})`,
          },
        )
      }
    } finally {
      setRun((r) => r && { ...r, streaming: false })
    }
  }

  function cite(n: number) {
    document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    setFlashed(null)
    requestAnimationFrame(() => setFlashed(n))
  }

  return (
    <div className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
      <section className={`transition-[padding] ${run ? 'pt-8' : 'pt-16 sm:pt-24'}`}>
        {!run && (
          <>
            <h1 className="max-w-[18ch] text-[clamp(2rem,5vw,3.25rem)] leading-[1.05] font-extrabold tracking-[-0.02em]">
              Ask what buyers think of a game or accessory
            </h1>
            <p className="mt-4 max-w-[60ch] text-lg text-ink-soft">
              Answers come from {meta ? fmtInt(meta.counts.products) : '4,000'} Amazon listings and their customer
              reviews, written through September 2023. Every claim links to the review or listing it came from.
            </p>
          </>
        )}

        <form
          className="mt-8 flex flex-col gap-2 sm:flex-row"
          onSubmit={(e) => {
            e.preventDefault()
            void submit(question)
          }}
        >
          <label htmlFor="q" className="sr-only">
            Your question
          </label>
          <input
            id="q"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. Which racing wheels work with PS5?"
            maxLength={500}
            autoComplete="off"
            className="h-13 w-full rounded-[10px] sm:flex-1 border border-line bg-surface px-4 text-base shadow-[0_1px_0_var(--color-line)] placeholder:text-ink-faint focus:border-action focus:outline-none"
          />
          <button
            type="submit"
            disabled={busy || question.trim().length < 3}
            className="h-13 rounded-[10px] bg-action px-6 font-semibold text-white hover:bg-action-ink disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? 'Answering' : 'Ask'}
          </button>
        </form>

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-ink-soft">
          <div className="flex items-center gap-2" role="radiogroup" aria-label="Search mode">
            <span>Search</span>
            <div className="flex rounded-md border border-line bg-surface p-0.5">
              {MODES.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  role="radio"
                  aria-checked={opts.mode === m.value}
                  onClick={() => setOpts({ ...opts, mode: m.value })}
                  className={`rounded px-2.5 py-1 text-xs font-semibold ${
                    opts.mode === m.value ? 'bg-ink text-white' : 'text-ink-soft hover:text-ink'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={opts.rerank}
              onChange={(e) => setOpts({ ...opts, rerank: e.target.checked })}
              className="h-4 w-4 accent-action"
            />
            Rerank results
          </label>
          {meta && meta.generators.length > 1 && (
            <label className="flex items-center gap-2">
              Model
              <select
                value={opts.generator ?? meta.default_generator}
                onChange={(e) => setOpts({ ...opts, generator: e.target.value })}
                className="rounded-md border border-line bg-surface px-2 py-1 text-xs"
              >
                {meta.generators.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>

        {!run && (
          <div className="mt-10">
            <p className="text-sm text-ink-faint">Try one of these</p>
            <ul className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <li key={ex}>
                  <button
                    type="button"
                    onClick={() => void submit(ex)}
                    className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm hover:border-ink-faint"
                  >
                    {ex}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {run && (
        <section className="mt-10 grid gap-8 lg:grid-cols-[minmax(0,1fr)_400px] lg:gap-10">
          <div>
            <h2 className="text-xl font-bold tracking-tight">{run.question}</h2>
            <div className="mt-4">
              {run.error ? (
                <p className="rounded-[10px] border border-star-1/40 bg-surface p-4 text-sm text-star-1">{run.error}</p>
              ) : run.text ? (
                <Answer
                  text={run.text}
                  streaming={run.streaming}
                  sources={run.sources}
                  onCite={cite}
                  onHover={setHovered}
                />
              ) : (
                <p className="text-ink-faint">
                  {!run.calls.length
                    ? 'Working out what to search for...'
                    : !run.sources.length
                      ? 'Searching listings and reviews...'
                      : `Reading ${run.sources.length} sources...`}
                </p>
              )}
            </div>
            <Trace calls={run.calls} timings={run.timings} model={run.model} usage={run.usage} settings={run.settings} />
          </div>
          <aside aria-label="Sources">
            <div className="lg:sticky lg:top-20">
              {run.sources.length > 0 && (
                <>
                  <div className="mb-3 flex items-baseline justify-between">
                    <h2 className="font-bold">Sources</h2>
                    <RatingKey />
                  </div>
                  <div className="lg:max-h-[calc(100vh-8rem)] lg:overflow-y-auto">
                    <Sources sources={run.sources} hovered={hovered} flashed={flashed} />
                  </div>
                </>
              )}
            </div>
          </aside>
        </section>
      )}
    </div>
  )
}

function RatingKey() {
  return (
    <div className="flex items-center gap-1.5 text-[11px] text-ink-faint" aria-label="Color shows the review's star rating">
      <span>1 star</span>
      {[1, 2, 3, 4, 5].map((r) => (
        <span key={r} className="h-2 w-3 rounded-sm" style={{ backgroundColor: `var(--color-star-${r})` }} />
      ))}
      <span>5 stars</span>
    </div>
  )
}
