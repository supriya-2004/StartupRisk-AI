// frontend/src/components/PipelineProgress.tsx

import { useEffect, useRef, useState } from 'react';
import { getPipelineStatus } from '../services/api';

interface PipelineProgressProps {
  threadId: string | null;
  isActive: boolean;
  timings?: Record<string, number>;
}

const STAGES = [
  { key: 'rag_retrieval',   label: 'RAG Retrieval',       icon: '🗂️',  timingKey: 'rag_retrieval' },
  { key: 'parallel_agents', label: 'LLM Agents',          icon: '🤖',  timingKey: 'parallel_agents' },
  { key: 'finbert',         label: 'FinBERT Sentiment',   icon: '🧠',  timingKey: 'finbert_sentiment' },
  { key: 'scoring',         label: 'Risk Scoring',        icon: '⚖️',  timingKey: 'score_calculation' },
  { key: 'synthesis',       label: 'Generative Synthesis',    icon: '✨',  timingKey: 'claude_synthesis' },
] as const;

type StageStatus = 'waiting' | 'running' | 'done';

export default function PipelineProgress({ threadId, isActive, timings: externalTimings }: PipelineProgressProps) {
  const [timings, setTimings] = useState<Record<string, number>>(externalTimings ?? {});
  const [startTime] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (externalTimings) setTimings(externalTimings);
  }, [externalTimings]);

  useEffect(() => {
    if (!isActive || !threadId) {
      if (pollRef.current) clearInterval(pollRef.current);
      if (tickRef.current) clearInterval(tickRef.current);
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const data = await getPipelineStatus(threadId);
        if (Object.keys(data).length > 0) setTimings(data);
      } catch {
        // ignore poll errors
      }
    }, 2000);

    tickRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, [isActive, threadId, startTime]);

  const getStatus = (timingKey: string, stageIndex: number): StageStatus => {
    if (timings[timingKey] !== undefined) return 'done';
    // Infer "running" if the previous stage is done but this one isn't
    const prevKey = stageIndex > 0 ? STAGES[stageIndex - 1].timingKey : null;
    if (stageIndex === 0 && isActive && Object.keys(timings).length === 0) return 'running';
    if (prevKey && timings[prevKey] !== undefined && timings[timingKey] === undefined && isActive) return 'running';
    return 'waiting';
  };

  const allDone = STAGES.every((s) => timings[s.timingKey] !== undefined);
  const totalElapsed = allDone
    ? Object.values(timings).reduce((a, b) => a + b, 0).toFixed(1)
    : null;

  return (
    <div className="pipeline-progress">
      <div className="pipeline-header">
        <span className="pipeline-title">⚡ Analysis Pipeline</span>
        {allDone && totalElapsed && (
          <span className="pipeline-total">Completed in {totalElapsed}s</span>
        )}
        {isActive && !allDone && (
          <span className="pipeline-elapsed">{elapsed}s elapsed…</span>
        )}
      </div>

      <div className="pipeline-stages">
        {STAGES.map((stage, i) => {
          const status = getStatus(stage.timingKey, i);
          const duration = timings[stage.timingKey];
          return (
            <div key={stage.key} className={`pipeline-stage pipeline-stage--${status}`}>
              <div className={`stage-dot ${status === 'running' ? 'stage-dot--pulse' : ''}`}>
                {status === 'done' ? '✓' : status === 'running' ? '…' : '○'}
              </div>
              <div className="stage-info">
                <span className="stage-icon">{stage.icon}</span>
                <span className="stage-label">{stage.label}</span>
                {duration !== undefined && (
                  <span className="stage-time">{duration.toFixed(2)}s</span>
                )}
              </div>
              {i < STAGES.length - 1 && <div className={`stage-connector ${status === 'done' ? 'stage-connector--done' : ''}`} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}
