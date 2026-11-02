export type ToolCall = { name: string; arguments: Record<string, unknown> }

export type Source = {
  n: number
  kind: 'product' | 'review'
  id: string
  parent_asin: string
  title: string
  text: string
  meta: {
    price?: number
    average_rating?: number
    rating_number?: number
    platform?: string
    product_type?: string
    store?: string
    rating?: number
    review_title?: string
    review_date?: string
    helpful_vote?: number
    verified_purchase?: boolean
  }
  score: number
  rerank_score: number | null
}

export type Timings = {
  route: number
  retrieve: number
  generate: number
  first_token: number
  total: number
}

export type AskEvent =
  | { type: 'plan'; calls: ToolCall[] }
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; text: string }
  | { type: 'done'; timings_ms: Timings; usage: { input_tokens: number; output_tokens: number }; model: string }
  | { type: 'error'; message: string }

export type Meta = {
  counts: { products: number; review_chunks: number }
  default_generator: string
  generators: string[]
  data_through: string
}

export type AskOptions = { generator?: string; mode: 'hybrid' | 'vector' | 'bm25'; rerank: boolean }

/** POST a question and call onEvent for each server-sent event as it arrives. */
export async function ask(question: string, opts: AskOptions, onEvent: (e: AskEvent) => void, signal?: AbortSignal) {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...opts }),
    signal,
  })
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '')
    throw new Error(`The server returned ${res.status}. ${detail}`.trim())
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const chunk = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const line = chunk.split('\n').find((l) => l.startsWith('data: '))
      if (line) onEvent(JSON.parse(line.slice(6)) as AskEvent)
    }
  }
}

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} returned ${res.status}`)
  return res.json() as Promise<T>
}
