import { useEffect, useState } from 'react'
import { getJSON } from '../api'
import { fmtMs } from '../lib'

type RetrievalRow = {
  n_strict: number
  'strict_hit@5': number
  'strict_hit@10': number
  strict_mrr: number
  n_lenient: number
  'lenient_hit@5': number
  lenient_mrr: number
  p50_latency_ms: number
}

type GenSummary = {
  generator: string
  n: number
  faithfulness: number
  correctness: number
  citation_quality: number
  abstain_rate_unanswerable: number
  false_abstain_rate_answerable: number
  invalid_citation_rate: number
  uncited_answer_rate: number
  price_constraint_ok: number | null
  p50_first_token_ms: number
  p50_total_ms: number
}

type Result = { kind: string; label: string; file: string; summary: unknown }

const CONFIG_LABEL: Record<string, string> = {
  bm25: 'Keyword (BM25)',
  vector: 'Vector',
  hybrid: 'Hybrid',
  'hybrid+rerank': 'Hybrid + rerank',
}

function Bar({ value, best }: { value: number; best: boolean }) {
  return (
    <div className="flex items-center justify-end gap-2">
      <span className={`num ${best ? 'font-bold text-ink' : 'text-ink-soft'}`}>{value.toFixed(2)}</span>
      <span className="relative hidden h-1.5 w-16 overflow-hidden rounded-full bg-ground sm:block">
        <span
          className={`absolute inset-y-0 left-0 rounded-full ${best ? 'bg-action' : 'bg-ink-faint'}`}
          style={{ width: `${value * 100}%` }}
        />
      </span>
    </div>
  )
}

function RetrievalTable({ summary }: { summary: Record<string, RetrievalRow> }) {
  const cols: { key: keyof RetrievalRow; label: string; hint: string }[] = [
    { key: 'strict_hit@5', label: 'Exact review in top 5', hint: 'strict hit@5' },
    { key: 'strict_hit@10', label: 'Exact review in top 10', hint: 'strict hit@10' },
    { key: 'strict_mrr', label: 'Rank of exact review', hint: 'strict MRR' },
    { key: 'lenient_hit@5', label: 'Right product in top 5', hint: 'lenient hit@5' },
  ]
  const rows = Object.entries(summary)
  const best = Object.fromEntries(cols.map((c) => [c.key, Math.max(...rows.map(([, r]) => r[c.key] as number))]))
  const first = rows[0]?.[1]

  return (
    <div className="overflow-x-auto rounded-[10px] border border-line bg-surface">
      <table className="w-full min-w-[640px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-faint">
            <th className="px-4 py-3 font-normal">First-stage search</th>
            {cols.map((c) => (
              <th key={c.key} className="px-4 py-3 text-right font-normal">
                <span className="block text-ink-soft">{c.label}</span>
                <span>{c.hint}</span>
              </th>
            ))}
            <th className="px-4 py-3 text-right font-normal">
              <span className="block text-ink-soft">Median time</span>
              <span>per query</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name} className="border-b border-line last:border-b-0">
              <td className="px-4 py-3 font-semibold">{CONFIG_LABEL[name] ?? name}</td>
              {cols.map((c) => (
                <td key={c.key} className="px-4 py-3">
                  <Bar value={r[c.key] as number} best={r[c.key] === best[c.key]} />
                </td>
              ))}
              <td className="num px-4 py-3 text-right text-ink-soft">{fmtMs(r.p50_latency_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {first && (
        <p className="border-t border-line px-4 py-3 text-xs text-ink-faint">
          Exact-review columns use {first.n_strict} questions written from a specific review. The right-product
          column uses {first.n_lenient}. MRR is 1 when the exact review ranks first, 0.5 when it ranks second, and so
          on.
        </p>
      )}
    </div>
  )
}

type RouterSummary = {
  generator: string
  n: number
  tool_accuracy: number
  n_with_filters: number
  filter_accuracy: number
  invented_filter_rate: number
  p50_latency_ms: number
}

function RouterTable({ runs }: { runs: RouterSummary[] }) {
  const rows: { label: string; get: (s: RouterSummary) => string }[] = [
    { label: 'Picked the expected search', get: (s) => pct(s.tool_accuracy) },
    { label: 'Extracted the stated price, platform, or sort', get: (s) => `${pct(s.filter_accuracy)} of ${s.n_with_filters}` },
    { label: 'Added a filter the golden set doesn\'t require', get: (s) => pct(s.invented_filter_rate) },
    { label: 'Median routing time', get: (s) => fmtMs(s.p50_latency_ms) },
  ]
  return (
    <div className="overflow-x-auto rounded-[10px] border border-line bg-surface">
      <table className="w-full min-w-[520px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-faint">
            <th className="px-4 py-3 font-normal">{runs[0]?.n} questions</th>
            {runs.map((r) => (
              <th key={r.generator} className="px-4 py-3 text-right font-semibold text-ink">
                {r.generator.replace(/^(ollama|anthropic):/, '')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-line last:border-b-0">
              <td className="px-4 py-2.5 text-ink-soft">{row.label}</td>
              {runs.map((r) => (
                <td key={r.generator} className="num px-4 py-2.5 text-right">
                  {row.get(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-line px-4 py-3 text-xs text-ink-faint">
        Extra filters are often correct, like "Switch" for a Switch screen protector, so the third row is an upper bound.
        When a platform or category filter leaves fewer than 3 results, the search retries without it, because Amazon
        mislabels some products.
      </p>
    </div>
  )
}

const pct = (x: number | null) => (x == null ? 'n/a' : `${Math.round(x * 100)}%`)
const score = (x: number) => (x ? `${x.toFixed(2)} / 5` : 'not judged')

function GenerationTable({ runs }: { runs: GenSummary[] }) {
  const rows: { label: string; get: (s: GenSummary) => string }[] = [
    { label: 'Faithfulness to sources (judge)', get: (s) => score(s.faithfulness) },
    { label: 'Correctness (judge)', get: (s) => score(s.correctness) },
    { label: 'Citation quality (judge)', get: (s) => score(s.citation_quality) },
    { label: 'Says "not in the data" when it isn\'t', get: (s) => (s.faithfulness ? pct(s.abstain_rate_unanswerable) : 'not judged') },
    { label: 'Answers with no citations', get: (s) => pct(s.uncited_answer_rate) },
    { label: 'Answers citing a source that doesn\'t exist', get: (s) => pct(s.invalid_citation_rate) },
    { label: 'Recommended products within the price limit', get: (s) => pct(s.price_constraint_ok) },
    { label: 'Median time to first word', get: (s) => fmtMs(s.p50_first_token_ms) },
    { label: 'Median time to full answer', get: (s) => fmtMs(s.p50_total_ms) },
  ]
  return (
    <div className="overflow-x-auto rounded-[10px] border border-line bg-surface">
      <table className="w-full min-w-[520px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-faint">
            <th className="px-4 py-3 font-normal">{runs[0]?.n} questions</th>
            {runs.map((r) => (
              <th key={r.generator} className="px-4 py-3 text-right font-semibold text-ink">
                {r.generator.replace(/^(ollama|anthropic):/, '')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-line last:border-b-0">
              <td className="px-4 py-2.5 text-ink-soft">{row.label}</td>
              {runs.map((r) => (
                <td key={r.generator} className="num px-4 py-2.5 text-right">
                  {row.get(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function EvalsPage() {
  const [results, setResults] = useState<Result[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getJSON<{ results: Result[] }>('/api/evals')
      .then((d) => setResults(d.results))
      .catch((e: Error) => setError(e.message))
  }, [])

  const retrieval = results?.find((r) => r.kind === 'retrieval')
  const generation = (results ?? []).filter((r) => r.kind === 'generation').map((r) => r.summary as GenSummary)
  const router = (results ?? []).filter((r) => r.kind === 'router').map((r) => r.summary as RouterSummary)

  return (
    <div className="mx-auto max-w-6xl px-4 pt-12 pb-24 sm:px-6">
      <h1 className="text-3xl font-extrabold tracking-[-0.02em]">How well it works</h1>
      <p className="mt-3 max-w-[64ch] text-ink-soft">
        Scores from the golden question set: questions with known answers, including some about products that aren't
        in the data. Rerun them with <code className="rounded bg-surface px-1 text-sm">uv run rra eval-retrieval</code>{' '}
        and <code className="rounded bg-surface px-1 text-sm">uv run rra eval-generation</code>.
      </p>

      {error && (
        <p className="mt-8 text-star-1">Couldn't load results from the backend ({error}). Start it with "uv run rra serve".</p>
      )}

      <section className="mt-10">
        <h2 className="text-lg font-bold">Finding the right evidence</h2>
        <p className="mt-1 mb-4 max-w-[64ch] text-sm text-ink-soft">
          Each search method gets the raw question. No LLM is involved, so these numbers are deterministic.
        </p>
        {retrieval ? (
          <RetrievalTable summary={retrieval.summary as Record<string, RetrievalRow>} />
        ) : (
          results && <p className="text-sm text-ink-faint">No retrieval results yet. Run uv run rra eval-retrieval.</p>
        )}
      </section>

      <section className="mt-12">
        <h2 className="text-lg font-bold">Choosing what to search</h2>
        <p className="mt-1 mb-4 max-w-[64ch] text-sm text-ink-soft">
          The model turns each question into tool calls: product search with filters, review search, or both.
        </p>
        {router.length ? (
          <RouterTable runs={router} />
        ) : (
          results && <p className="text-sm text-ink-faint">No router results yet. Run uv run rra eval-router.</p>
        )}
      </section>

      <section className="mt-12">
        <h2 className="text-lg font-bold">Writing the answer</h2>
        <p className="mt-1 mb-4 max-w-[64ch] text-sm text-ink-soft">
          The full pipeline answers every question. Code checks citations and price limits. A Claude judge scores
          faithfulness and correctness against the sources each model saw.
        </p>
        {generation.length ? (
          <GenerationTable runs={generation} />
        ) : (
          results && <p className="text-sm text-ink-faint">No generation results yet. Run uv run rra eval-generation.</p>
        )}
      </section>
    </div>
  )
}
