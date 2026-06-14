-- Owner-controlled service editor. Existing service rows are preserved.

alter table public.services
    add column if not exists category_key text,
    add column if not exists variant_name text,
    add column if not exists description text,
    add column if not exists duration_minutes integer,
    add column if not exists requires_duration_choice boolean,
    add column if not exists sort_order integer;

update public.services
set category_key = coalesce(category_key, category),
    duration_minutes = coalesce(duration_minutes, fixed_duration_minutes),
    requires_duration_choice = coalesce(
        requires_duration_choice,
        booking_mode = 'choose_duration'
    ),
    sort_order = coalesce(sort_order, display_order, 0)
where category_key is null
   or requires_duration_choice is null
   or sort_order is null;

alter table public.services
    alter column category_key set default 'other',
    alter column requires_duration_choice set default false,
    alter column sort_order set default 0;

alter table public.services
    drop constraint if exists services_category_key_format,
    add constraint services_category_key_format
        check (category_key ~ '^[a-z0-9_]+$'),
    drop constraint if exists services_service_name_length,
    add constraint services_service_name_length
        check (char_length(btrim(service_name)) between 2 and 80),
    drop constraint if exists services_variant_name_length,
    add constraint services_variant_name_length
        check (variant_name is null or char_length(btrim(variant_name)) <= 80),
    drop constraint if exists services_description_length,
    add constraint services_description_length
        check (description is null or char_length(btrim(description)) <= 500),
    drop constraint if exists services_duration_minutes_range,
    add constraint services_duration_minutes_range
        check (
            requires_duration_choice = true
            or duration_minutes between 5 and 300
        ),
    drop constraint if exists services_price_pence_range,
    add constraint services_price_pence_range
        check (price_pence is null or price_pence between 0 and 100000),
    drop constraint if exists services_sort_order_range,
    add constraint services_sort_order_range
        check (sort_order between 0 and 9999);

create index if not exists services_owner_editor_sort_idx
    on public.services (business_slug, category_key, sort_order, service_name);

alter table public.services enable row level security;

revoke all on public.services from anon;
grant select, insert, update on public.services to authenticated;

drop policy if exists "Authenticated owner can view services" on public.services;
create policy "Authenticated owner can view services"
    on public.services for select to authenticated using (true);

drop policy if exists "Authenticated owner can insert services" on public.services;
create policy "Authenticated owner can insert services"
    on public.services for insert to authenticated with check (true);

drop policy if exists "Authenticated owner can update services" on public.services;
create policy "Authenticated owner can update services"
    on public.services for update to authenticated using (true) with check (true);
