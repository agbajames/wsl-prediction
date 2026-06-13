-- scripts/setup_prediction_runs_table.sql
-- -----------------------------------------------------------------------
-- Run this once in your Supabase SQL editor to create the audit table
-- that logs every prediction run from the API.
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.prediction_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Date window
    train_before        DATE NOT NULL,
    predict_from        DATE NOT NULL,
    predict_to          DATE NOT NULL,

    -- Model configuration used for this run
    model_config        JSONB NOT NULL,

    -- Outputs
    predictions         JSONB NOT NULL,   -- array of prediction objects
    team_strengths      JSONB,            -- array of team strength objects
    rho_fitted          FLOAT,            -- actual rho used (fitted or fixed)

    -- Backtest metrics (populated if backtest was run alongside prediction)
    brier_score         FLOAT,
    log_loss            FLOAT,
    n_matches_eval      INTEGER,

    -- Audit
    run_trigger         TEXT DEFAULT 'api'  -- 'api' | 'scheduled' | 'manual'
);

-- Index for fast recent-run queries
CREATE INDEX IF NOT EXISTS idx_prediction_runs_created
    ON public.prediction_runs (created_at DESC);

-- Index for querying by prediction window
CREATE INDEX IF NOT EXISTS idx_prediction_runs_window
    ON public.prediction_runs (predict_from, predict_to);

-- RLS: only authenticated service role can read/write
ALTER TABLE public.prediction_runs ENABLE ROW LEVEL SECURITY;

-- Allow the API (service role) full access
-- Note: service role bypasses RLS by default in Supabase.
-- Add explicit policies only if you want anon/authenticated roles to read:

-- Example: allow authenticated users to read prediction history
-- CREATE POLICY "Authenticated users can read predictions"
--     ON public.prediction_runs FOR SELECT
--     TO authenticated
--     USING (true);

COMMENT ON TABLE public.prediction_runs IS
    'Audit log of every WSL prediction engine run. '
    'Populated automatically by the FastAPI /predict endpoint.';
