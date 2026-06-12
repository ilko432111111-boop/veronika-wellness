-- Temporary Google Calendar events for active SumUp deposit holds.
-- Run this migration manually in the Supabase SQL editor.

alter table public.payment_requests
    add column if not exists calendar_hold_event_id text null,
    add column if not exists calendar_hold_created_at timestamptz null,
    add column if not exists calendar_hold_deleted_at timestamptz null,
    add column if not exists calendar_hold_error text null,
    add column if not exists calendar_hold_status text null;

alter table public.payment_requests
    drop constraint if exists payment_requests_calendar_hold_status_check;

alter table public.payment_requests
    add constraint payment_requests_calendar_hold_status_check check (
        calendar_hold_status is null
        or calendar_hold_status in (
            'creating',
            'active',
            'deleted',
            'converted',
            'failed'
        )
    );

update public.payment_requests
set calendar_hold_status = 'failed',
    calendar_hold_error = 'Temporary Google Calendar hold has not been added yet.',
    updated_at = now()
where status = 'pending'
  and hold_expires_at > now()
  and calendar_hold_event_id is null
  and calendar_hold_status is null;

create index if not exists payment_requests_calendar_hold_cleanup_idx
    on public.payment_requests (calendar_hold_status, hold_expires_at)
    where calendar_hold_event_id is not null;
