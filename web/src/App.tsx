import { useEffect, useState } from 'react'
import { getJSON, type Meta } from './api'
import { fmtInt } from './lib'
import { AskPage } from './pages/Ask'
import { EvalsPage } from './pages/Evals'

function useHashRoute() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash === '#/evals' ? 'evals' : 'ask'
}

function Mark() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
      <rect width="32" height="32" rx="7" fill="var(--color-ink)" />
      <g fill="none" strokeWidth="3.2" strokeLinecap="round">
        <path d="M9 22v-4" stroke="var(--color-star-1)" />
        <path d="M13.5 22v-7" stroke="var(--color-star-2)" />
        <path d="M18 22v-9" stroke="var(--color-star-3)" />
        <path d="M22.5 22v-12" stroke="var(--color-star-5)" />
      </g>
    </svg>
  )
}

export default function App() {
  const route = useHashRoute()
  const [meta, setMeta] = useState<Meta | null>(null)

  useEffect(() => {
    getJSON<Meta>('/api/meta').then(setMeta).catch(() => setMeta(null))
  }, [])

  const tab = (id: string, href: string, label: string) => (
    <a
      href={href}
      aria-current={route === id ? 'page' : undefined}
      className={`rounded-md px-3 py-1.5 text-sm font-semibold ${
        route === id ? 'bg-ground text-ink' : 'text-ink-soft hover:text-ink'
      }`}
    >
      {label}
    </a>
  )

  return (
    <>
      <header className="sticky top-0 z-10 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6">
          <a href="#/" className="flex items-center gap-2.5 font-bold tracking-tight">
            <Mark />
            <span className="hidden sm:inline">Review RAG Assistant</span>
          </a>
          <nav className="flex gap-1">
            {tab('ask', '#/', 'Ask')}
            {tab('evals', '#/evals', 'Evals')}
          </nav>
          {meta && (
            <p className="num ml-auto hidden text-xs text-ink-faint md:block">
              {fmtInt(meta.counts.products)} products, {fmtInt(meta.counts.review_chunks)} review passages, data
              through Sep 2023
            </p>
          )}
        </div>
      </header>
      <main>{route === 'evals' ? <EvalsPage /> : <AskPage meta={meta} />}</main>
    </>
  )
}
