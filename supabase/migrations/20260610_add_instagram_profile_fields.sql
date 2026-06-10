alter table public.channel_contacts
    add column if not exists profile_name text,
    add column if not exists profile_username text,
    add column if not exists profile_fetched_at timestamptz;
