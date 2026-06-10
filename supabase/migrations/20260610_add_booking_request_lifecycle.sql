-- Run this migration in the Supabase SQL editor, or apply it with the
-- Supabase CLI. It preserves all historical booking request rows.

alter table public.booking_requests
    add column if not exists calendar_event_id text,
    add column if not exists confirmed_at timestamptz,
    add column if not exists confirmation_lock_token text,
    add column if not exists confirmation_started_at timestamptz;

create index if not exists booking_requests_lifecycle_status_idx
    on public.booking_requests (request_status, preferred_date, preferred_time);

create or replace function public.claim_booking_request_confirmation(
    target_request_id bigint,
    new_lock_token text
)
returns setof public.booking_requests
language plpgsql
security definer
set search_path = public
as $$
begin
    return query
    update public.booking_requests
    set confirmation_lock_token = new_lock_token,
        confirmation_started_at = now(),
        updated_at = now()
    where id = target_request_id
      and request_status in ('draft', 'ready_for_review', 'awaiting_owner_confirmation')
      and calendar_event_id is null
      and (
          confirmation_lock_token is null
          or confirmation_started_at < now() - interval '10 minutes'
      )
    returning *;
end;
$$;

create or replace function public.release_booking_request_confirmation(
    target_request_id bigint,
    lock_token text
)
returns void
language sql
security definer
set search_path = public
as $$
    update public.booking_requests
    set confirmation_lock_token = null,
        confirmation_started_at = null,
        updated_at = now()
    where id = target_request_id
      and confirmation_lock_token = lock_token
      and calendar_event_id is null;
$$;

revoke all on function public.claim_booking_request_confirmation(bigint, text) from public;
revoke all on function public.release_booking_request_confirmation(bigint, text) from public;
grant execute on function public.claim_booking_request_confirmation(bigint, text) to service_role;
grant execute on function public.release_booking_request_confirmation(bigint, text) to service_role;

create or replace function public.expire_stale_booking_requests()
returns table (
    expired_count bigint,
    completed_count bigint,
    abandoned_count bigint
)
language plpgsql
security definer
set search_path = public
as $$
declare
    -- One clear configuration value for incomplete request inactivity.
    abandoned_after interval := interval '72 hours';
    expired_rows bigint := 0;
    completed_rows bigint := 0;
    abandoned_rows bigint := 0;
begin
    with expired as (
        update public.booking_requests
        set request_status = 'expired',
            confirmation_lock_token = null,
            confirmation_started_at = null,
            updated_at = now()
        where request_status in ('draft', 'ready_for_review', 'awaiting_owner_confirmation')
          and preferred_date is not null
          and preferred_time is not null
          and (
              (preferred_date::text || ' ' || preferred_time::text)::timestamp
              at time zone 'Europe/London'
          ) < now()
        returning id
    )
    select count(*) into expired_rows from expired;

    with completed as (
        update public.booking_requests
        set request_status = 'completed',
            updated_at = now()
        where request_status = 'confirmed'
          and preferred_date is not null
          and preferred_time is not null
          and total_duration_minutes is not null
          and (
              (
                  (preferred_date::text || ' ' || preferred_time::text)::timestamp
                  at time zone 'Europe/London'
              )
              + make_interval(mins => total_duration_minutes)
          ) < now()
        returning id
    )
    select count(*) into completed_rows from completed;

    with abandoned as (
        update public.booking_requests
        set request_status = 'abandoned',
            confirmation_lock_token = null,
            confirmation_started_at = null,
            updated_at = now()
        where request_status = 'draft'
          and updated_at < now() - abandoned_after
          and (
              preferred_date is null
              or preferred_time is null
              or total_duration_minutes is null
              or missing_detail is not null
          )
        returning id
    )
    select count(*) into abandoned_rows from abandoned;

    update public.conversations as conversation
    set active_booking_request_id = null,
        updated_at = now()
    from public.booking_requests as request
    where conversation.active_booking_request_id = request.id
      and request.request_status in (
          'expired',
          'completed',
          'cancelled_by_customer',
          'abandoned'
      );

    return query
    select expired_rows, completed_rows, abandoned_rows;
end;
$$;

revoke all on function public.expire_stale_booking_requests() from public;
grant execute on function public.expire_stale_booking_requests() to service_role;

create extension if not exists pg_cron with schema extensions;

do $$
declare
    existing_job_id bigint;
begin
    select jobid
    into existing_job_id
    from cron.job
    where jobname = 'expire-stale-booking-requests'
    limit 1;

    if existing_job_id is not null then
        perform cron.unschedule(existing_job_id);
    end if;

    perform cron.schedule(
        'expire-stale-booking-requests',
        '*/15 * * * *',
        'select public.expire_stale_booking_requests();'
    );
end;
$$;
