-- scripts/setup_evaluation_runs_table.sql
-- -----------------------------------------------------------------------
-- Run this once in your Supabase SQL editor to create the table that logs
-- offline evaluation and backtest runs.
--
-- This table is separate from prediction_runs by design. Prediction run
-- logging remains backward compatible and untouched.
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.evaluation_runs (
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Evaluation scope
    evaluation_type     TEXT NOT NULL,
    start_date          DATE NOT NULL,
    end_date            DATE,

    -- Configuration and parameters
    model_config        JSONB NOT NULL,
    evaluation_params   JSONB NOT NULL,

    -- Results
    aggregate_metrics   JSONB NOT NULL,
    calibration_bins    JSONB,
    confidence_buckets  JSONB,
    per_match_results   JSONB,

    -- Data and audit metadata
    data_snapshot       JSONB,
    run_trigger         TEXT NOT NULL DEFAULT 'manual',
    code_version        TEXT,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created
    ON public.evaluation_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_type_created
    ON public.evaluation_runs (evaluation_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_window
    ON public.evaluation_runs (start_date, end_date);

ALTER TABLE public.evaluation_runs ENABLE ROW LEVEL SECURITY;

-- The API/service role bypasses RLS by default in Supabase.
-- Add explicit SELECT policies only if non-service roles should read results.

COMMENT ON TABLE public.evaluation_runs IS
    'Audit log of offline WSL prediction engine evaluation and backtest runs.';
