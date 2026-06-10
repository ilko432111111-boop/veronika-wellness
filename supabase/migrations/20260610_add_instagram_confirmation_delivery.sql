create table if not exists public.channel_contacts (
    id uuid primary key default gen_random_uuid(),
    business_slug text not null,
    source text not null,
    external_user_id text not null,
    last_customer_message_at timestamptz null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (business_slug, source, external_user_id)
);

alter table public.conversations
    add column if not exists channel_contact_id uuid
        references public.channel_contacts(id) on delete set null;

create index if not exists conversations_channel_contact_id_idx
    on public.conversations (channel_contact_id);

alter table public.booking_requests
    add column if not exists confirmation_message_status text
        not null default 'not_attempted',
    add column if not exists confirmation_message_sent_at timestamptz null,
    add column if not exists confirmation_message_error text null;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'booking_requests_confirmation_message_status_check'
    ) then
        alter table public.booking_requests
            add constraint booking_requests_confirmation_message_status_check
            check (
                confirmation_message_status in (
                    'sent',
                    'skipped_non_instagram',
                    'skipped_reply_window_expired',
                    'failed',
                    'not_attempted'
                )
            );
    end if;
end;
$$;

alter table public.channel_contacts enable row level security;

drop policy if exists "Authenticated admins can view channel contacts"
    on public.channel_contacts;

create policy "Authenticated admins can view channel contacts"
    on public.channel_contacts
    for select
    to authenticated
    using (true);

grant select on public.channel_contacts to authenticated;
