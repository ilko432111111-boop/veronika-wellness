alter table public.conversations
add column if not exists phone_declined boolean not null default false;
