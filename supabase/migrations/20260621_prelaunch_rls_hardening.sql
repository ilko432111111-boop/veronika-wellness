-- Pre-launch privacy hardening for customer, owner, payment, and token data.
-- Keeps public website access through the FastAPI server while preventing
-- direct anonymous Supabase reads/writes of private tables.

do $$
declare
    table_name text;
    private_tables text[] := array[
        'conversations',
        'messages',
        'booking_requests',
        'booking_items',
        'channel_contacts',
        'payment_requests',
        'pending_booking_actions',
        'chatbot_settings',
        'business_hours',
        'services',
        'documents',
        'google_calendar_tokens'
    ];
begin
    foreach table_name in array private_tables loop
        if to_regclass(format('public.%I', table_name)) is not null then
            execute format('alter table public.%I enable row level security', table_name);
            execute format('revoke all on public.%I from anon', table_name);
        end if;
    end loop;

    if to_regclass('public.google_calendar_tokens') is not null then
        execute 'revoke all on public.google_calendar_tokens from authenticated';
    end if;

    if to_regclass('public.documents') is not null then
        execute 'revoke all on public.documents from authenticated';
    end if;
end;
$$;

do $$
begin
    if to_regclass('public.conversations') is not null then
        grant select, update on public.conversations to authenticated;
        drop policy if exists "Authenticated owner can view conversations" on public.conversations;
        create policy "Authenticated owner can view conversations"
            on public.conversations
            for select
            to authenticated
            using (true);

        drop policy if exists "Authenticated owner can update conversations" on public.conversations;
        create policy "Authenticated owner can update conversations"
            on public.conversations
            for update
            to authenticated
            using (true)
            with check (true);
    end if;

    if to_regclass('public.messages') is not null then
        grant select on public.messages to authenticated;
        drop policy if exists "Authenticated owner can view messages" on public.messages;
        create policy "Authenticated owner can view messages"
            on public.messages
            for select
            to authenticated
            using (true);
    end if;

    if to_regclass('public.booking_requests') is not null then
        grant select on public.booking_requests to authenticated;
        drop policy if exists "Authenticated owner can view booking requests" on public.booking_requests;
        create policy "Authenticated owner can view booking requests"
            on public.booking_requests
            for select
            to authenticated
            using (true);
    end if;

    if to_regclass('public.booking_items') is not null then
        grant select on public.booking_items to authenticated;
        drop policy if exists "Authenticated owner can view booking items" on public.booking_items;
        create policy "Authenticated owner can view booking items"
            on public.booking_items
            for select
            to authenticated
            using (true);
    end if;

    if to_regclass('public.channel_contacts') is not null then
        grant select on public.channel_contacts to authenticated;
        drop policy if exists "Authenticated admins can view channel contacts" on public.channel_contacts;
        create policy "Authenticated admins can view channel contacts"
            on public.channel_contacts
            for select
            to authenticated
            using (true);
    end if;

    if to_regclass('public.payment_requests') is not null then
        grant select on public.payment_requests to authenticated;
        drop policy if exists "Authenticated admins can view payment requests" on public.payment_requests;
        create policy "Authenticated admins can view payment requests"
            on public.payment_requests
            for select
            to authenticated
            using (true);
    end if;

    if to_regclass('public.pending_booking_actions') is not null then
        grant select on public.pending_booking_actions to authenticated;
        drop policy if exists "Authenticated owner can view pending booking actions" on public.pending_booking_actions;
        create policy "Authenticated owner can view pending booking actions"
            on public.pending_booking_actions
            for select
            to authenticated
            using (true);
    end if;

    if to_regclass('public.chatbot_settings') is not null then
        grant select, update on public.chatbot_settings to authenticated;
        drop policy if exists "Authenticated owner can view chatbot settings" on public.chatbot_settings;
        create policy "Authenticated owner can view chatbot settings"
            on public.chatbot_settings
            for select
            to authenticated
            using (true);

        drop policy if exists "Authenticated owner can update chatbot settings" on public.chatbot_settings;
        create policy "Authenticated owner can update chatbot settings"
            on public.chatbot_settings
            for update
            to authenticated
            using (true)
            with check (true);
    end if;

    if to_regclass('public.business_hours') is not null then
        grant select, update on public.business_hours to authenticated;
        drop policy if exists "Authenticated owner can view business hours" on public.business_hours;
        create policy "Authenticated owner can view business hours"
            on public.business_hours
            for select
            to authenticated
            using (true);

        drop policy if exists "Authenticated owner can update business hours" on public.business_hours;
        create policy "Authenticated owner can update business hours"
            on public.business_hours
            for update
            to authenticated
            using (true)
            with check (true);
    end if;
end;
$$;
