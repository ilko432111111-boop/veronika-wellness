-- Stage 1 SumUp deposits: payment links and temporary slot holds only.
-- This migration is non-destructive and preserves all booking/payment history.

create table if not exists public.payment_requests (
    id uuid primary key default gen_random_uuid(),
    business_slug text not null,
    booking_request_id bigint not null
        references public.booking_requests(id) on delete cascade,
    provider text not null default 'sumup',
    payment_type text not null default 'deposit',
    amount_pence integer not null,
    currency text not null default 'GBP',
    provider_checkout_id text null,
    checkout_reference text not null unique,
    hosted_checkout_url text null,
    status text not null default 'pending',
    held_start_at timestamptz not null,
    held_end_at timestamptz not null,
    hold_expires_at timestamptz not null,
    paid_at timestamptz null,
    expired_at timestamptz null,
    cancelled_at timestamptz null,
    provider_error text null,
    message_delivery_status text not null default 'not_attempted',
    message_delivery_error text null,
    message_sent_at timestamptz null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint payment_requests_amount_positive check (amount_pence > 0),
    constraint payment_requests_hold_interval_valid check (
        held_end_at > held_start_at
    ),
    constraint payment_requests_status_check check (
        status in ('pending', 'paid', 'failed', 'expired', 'cancelled')
    ),
    constraint payment_requests_delivery_status_check check (
        message_delivery_status in (
            'not_attempted',
            'sent',
            'failed',
            'reply_window_expired',
            'not_applicable'
        )
    )
);

create index if not exists payment_requests_booking_request_id_idx
    on public.payment_requests (booking_request_id);

create index if not exists payment_requests_status_idx
    on public.payment_requests (status);

create index if not exists payment_requests_hold_expires_at_idx
    on public.payment_requests (hold_expires_at);

create index if not exists payment_requests_held_interval_idx
    on public.payment_requests (held_start_at, held_end_at);

create index if not exists payment_requests_checkout_reference_idx
    on public.payment_requests (checkout_reference);

create index if not exists payment_requests_provider_checkout_id_idx
    on public.payment_requests (provider_checkout_id);

create unique index if not exists payment_requests_one_pending_per_booking_idx
    on public.payment_requests (booking_request_id)
    where status = 'pending';

alter table public.payment_requests enable row level security;

drop policy if exists "Authenticated admins can view payment requests"
    on public.payment_requests;

create policy "Authenticated admins can view payment requests"
    on public.payment_requests
    for select
    to authenticated
    using (true);

grant select on public.payment_requests to authenticated;

create or replace function public.expire_stale_payment_requests()
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    expired_rows bigint := 0;
begin
    with expired as (
        update public.payment_requests
        set status = 'expired',
            expired_at = coalesce(expired_at, now()),
            updated_at = now()
        where status = 'pending'
          and hold_expires_at <= now()
        returning id
    )
    select count(*) into expired_rows from expired;

    return expired_rows;
end;
$$;

revoke all on function public.expire_stale_payment_requests() from public;
grant execute on function public.expire_stale_payment_requests() to service_role;

create extension if not exists pg_cron with schema extensions;

do $$
declare
    existing_job_id bigint;
begin
    select jobid
    into existing_job_id
    from cron.job
    where jobname = 'expire-stale-payment-requests'
    limit 1;

    if existing_job_id is not null then
        perform cron.unschedule(existing_job_id);
    end if;

    perform cron.schedule(
        'expire-stale-payment-requests',
        '*/5 * * * *',
        'select public.expire_stale_payment_requests();'
    );
end;
$$;
