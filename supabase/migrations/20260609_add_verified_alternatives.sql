alter table public.conversations
add column if not exists verified_alternatives jsonb not null default '[]'::jsonb;
