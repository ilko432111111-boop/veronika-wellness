alter table public.conversations
    add column if not exists internal_notes text;

do $$
begin
    if exists (
        select 1
        from information_schema.columns
        where table_schema = 'public'
          and table_name = 'conversations'
          and column_name = 'channel_contact_id'
    ) then
        create index if not exists conversations_channel_contact_id_idx
            on public.conversations (channel_contact_id);
    end if;
end;
$$;

alter table public.conversations enable row level security;

revoke all on public.conversations from anon;
grant select, update on public.conversations to authenticated;

drop policy if exists "Authenticated owner can view conversations"
    on public.conversations;

create policy "Authenticated owner can view conversations"
    on public.conversations
    for select
    to authenticated
    using (true);

drop policy if exists "Authenticated owner can update conversations"
    on public.conversations;

create policy "Authenticated owner can update conversations"
    on public.conversations
    for update
    to authenticated
    using (true)
    with check (true);
